"""Solar Open3 SDA (grouped-decay delta rule, exact erase) on ROCm: the SGLang side.

The prefill kernel itself is aiter's ``flash_sda_fwd`` and is tested in aiter
(``op_tests/triton_tests/chunk_delta_attn/test_flash_sda.py``). This file covers what
SGLang adds around it: the adapter that replaces the Triton ``chunk_kda`` global
(called through SGLang's own ``TritonKDAKernel.extend``), the hand-off to the stock
Triton decode / target-verify kernel with the exact-erase beta cap -- the three
serving stages of the model (prefill / extend, decode, mixed batches) -- and the
model wiring in ``solar_open3.py``.

Reference (``_ref``), per head:
    kn = k * rsqrt(|k|^2 + 1e-6), qn likewise        (fp64)
    b  = min(beta, 2 - 2^-6)                            (exact erase)
    g  = -exp(A_log) * softplus(g_raw[::16] + dt_bias[::16])   (8 groups)
    S  = (I - b kn kn^T) Diag(exp g) S + b kn v^T,  o = S^T qn / sqrt(128)
"""

import math
import unittest

import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=240, suite="stage-a-test-1-gpu-small-amd")

from sglang.srt.layers.attention.solar_open3.amd import sglang_sda_rocm as ADP

DEV = "cuda"
H, K, G = 24, 128, 8
GW = K // G
BETA_MAX = 2.0 - 2.0**-6
SCALE = K**-0.5
BF = torch.bfloat16


def _skip_reason():
    if not torch.cuda.is_available():
        return "no GPU"
    if torch.version.hip is None:
        return "ROCm-only kernel"
    return None


SKIP = _skip_reason()


# ---------------------------------------------------------------- reference
def _gate(g_raw, A_log, dt_bias):
    z = g_raw[..., ::GW].double()
    if dt_bias is not None:
        z = z + dt_bias.reshape(H, K)[:, ::GW].double()
    return -A_log.reshape(H, 1).double().exp() * F.softplus(z)


def _ref(q, k, v, g_raw, beta, A_log, dt_bias, cu, h0=None, exact=True):
    """fp64 recurrence. h0 [N, H, V, K] (SGLang's V-first pool layout) or None.
    Returns o [1, T, H, V], finals [N, H, V, K], snaps[n][c64] = state before
    64-chunk c64 ([H, V, K])."""
    qd, kd, vd = q[0].double(), k[0].double(), v[0].double()
    qn = qd * torch.rsqrt((qd * qd).sum(-1, keepdim=True) + 1e-6)
    kn = kd * torch.rsqrt((kd * kd).sum(-1, keepdim=True) + 1e-6)
    b = beta[0].double()
    if exact:
        b = b.clamp(max=BETA_MAX)
    alpha = _gate(g_raw[0], A_log, dt_bias).exp().repeat_interleave(GW, -1)
    o = torch.zeros_like(vd)
    finals, snaps = [], []
    cu = cu.tolist()
    for n in range(len(cu) - 1):
        S = (h0[n].double().transpose(-1, -2).clone() if h0 is not None
             else torch.zeros(H, K, K, dtype=torch.float64, device=q.device))
        snap = {}
        for t in range(cu[n], cu[n + 1]):
            if (t - cu[n]) % 64 == 0:
                snap[(t - cu[n]) // 64] = S.transpose(-1, -2).clone()
            S = S * alpha[t].unsqueeze(-1)
            err = vd[t] - torch.einsum("hk,hkv->hv", kn[t], S)
            S = S + torch.einsum("hk,hv->hkv", b[t].unsqueeze(-1) * kn[t], err)
            o[t] = SCALE * torch.einsum("hk,hkv->hv", qn[t], S)
        finals.append(S.transpose(-1, -2))
        snaps.append(snap)
    return o.unsqueeze(0), torch.stack(finals), snaps


# ---------------------------------------------------------------- inputs
def _params(seed=0):
    torch.manual_seed(seed)
    # Ranges of the trained checkpoint (exp(A_log) 1.2 .. 36, dt_bias -7.5 .. -1.7).
    A_log = torch.empty(H, device=DEV).uniform_(0.2, 3.5)
    dt_bias = torch.empty(H, G, device=DEV).uniform_(-7.5, -1.7)
    dt_bias = dt_bias.repeat_interleave(GW, -1).reshape(-1)
    return A_log, dt_bias


def _inputs(lens, seed=0, zmu=0.0, zsig=1.5, blogit=0.0, corr=0.0, strided=True):
    """q/k/v as views of one packed qkv (as the KDA backend hands them over),
    group-constant raw gate, activated beta (2 sigmoid)."""
    torch.manual_seed(seed)
    T = sum(lens)
    qkv = torch.randn(T, 3 * H * K, device=DEV)
    if corr:
        base = torch.randn(1, H * K, device=DEV)
        kk = qkv[:, H * K: 2 * H * K]
        kk.mul_(1 - corr).add_(corr * base)
    qkv = qkv.to(BF)
    q, k, v = (x.unflatten(-1, (H, K)).unsqueeze(0) for x in qkv.split(H * K, -1))
    if not strided:
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    gz = zmu + torch.randn(1, T, H, G, device=DEV) * zsig
    g = gz.repeat_interleave(GW, -1).to(BF)
    beta = 2 * torch.sigmoid(blogit + torch.randn(1, T, H, device=DEV) * 2)
    cu = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), device=DEV,
                      dtype=torch.int32)
    return q, k, v, g, beta, cu


def _rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


def _pool(n_slots, seed=1, scale=0.2):
    torch.manual_seed(seed)
    return torch.randn(n_slots, H, K, K, device=DEV) * scale  # [slots, H, V, K]


# Tolerances: the kernel's operands are bf16 (as in FlashSDA and fla); the
# output is rounded to bf16 (relative 2^-9 alone).
TOL_O = 1e-2
TOL_S = 1e-2


@unittest.skipIf(SKIP, SKIP)
class TestSdaServingStages(unittest.TestCase):
    """Prefill (``SdaRocmPrefill.extend`` with the KDA backend's arguments, and the
    backend's own ``forward_extend`` routing an SDA layer to it), decode and
    target-verify (SGLang's Triton recurrent kernel with the capped logit), and mixed
    batches (decode rows riding the chunked prefill)."""

    @classmethod
    def setUpClass(cls):
        from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel

        cls.kernel = TritonKDAKernel()  # decode / verify
        cls.cap = ADP.beta_logit_cap(2.0)

    def setUp(self):
        self.A_log, self.dt_bias = _params()
        self.sda = ADP.SdaRocmPrefill(groups=G, exact_erase=True)

    def _extend(self, q, k, v, g, beta, cu, pool, idx, **kw):
        # The keyword set KDAAttnBackend.forward_extend hands its prefill kernel.
        return self.sda.extend(
            q=q, k=k, v=v, g=g, beta=beta, ssm_states=pool, cache_indices=idx,
            query_start_loc=cu, A_log=self.A_log.view(1, 1, H, 1), dt_bias=self.dt_bias,
            lower_bound=None, beta_is_raw=False, beta_scale=2.0,
            extend_seq_lens_cpu=None, extend_prefix_lens=None, layer_id=0,
            is_spec_decode=False, **kw)

    def _decode(self, q, k, v, g, blogit, pool, idx):
        B = q.shape[1]
        qsl = torch.arange(B + 1, device=DEV, dtype=torch.int32)
        return self.kernel.decode(
            q, k, v, g[0].reshape(B, H * K), blogit, A_log=self.A_log.view(1, 1, H, 1),
            dt_bias=self.dt_bias, ssm_states=pool, cache_indices=idx,
            query_start_loc=qsl, lower_bound=None, beta_scale=2.0)

    def test_beta_logit_cap(self):
        c = torch.tensor([self.cap], device=DEV)
        self.assertGreaterEqual(float(2 * torch.sigmoid(c)), BETA_MAX)
        below = torch.nextafter(c, torch.tensor([-math.inf], device=DEV))
        self.assertLess(float(2 * torch.sigmoid(below)), BETA_MAX + 2e-6)
        self.assertLess(float(2 * torch.sigmoid(c)) - BETA_MAX, 1e-6)

    def test_prefill_through_sglang_extend(self):
        lens = [100, 1, 257]
        q, k, v, g, beta, cu = _inputs(lens)
        pool = _pool(5)
        idx = torch.tensor([3, 1, 4], device=DEV, dtype=torch.int32)
        h0 = pool[idx.long()].clone()
        o = self._extend(q, k, v, g, beta, cu, pool, idx)
        ro, rf, _ = _ref(q, k, v, g, beta, self.A_log, self.dt_bias, cu, h0)
        self.assertLess(_rel(o, ro), TOL_O)
        self.assertLess(_rel(pool[idx.long()], rf), TOL_S)

    def test_prefill_with_track_through_sglang_extend(self):
        lens = [150, 70]
        q, k, v, g, beta, cu = _inputs(lens)
        pool = _pool(3)
        idx = torch.tensor([1, 2], device=DEV, dtype=torch.int32)
        h0 = pool[idx.long()].clone()
        track = torch.empty(2, H, K, K, device=DEV)
        tci = torch.tensor([2, 1], device=DEV, dtype=torch.int32)
        o, h = self._extend(q, k, v, g, beta, cu, pool, idx, return_intermediate_states=True,
                            track_state=track, track_chunk_idx=tci)
        self.assertIsNone(h)  # the backend reads the snapshot, never h, in this case
        _, _, snaps = _ref(q, k, v, g, beta, self.A_log, self.dt_bias, cu, h0)
        self.assertLess(_rel(track[0], snaps[0][2]), TOL_S)
        self.assertLess(_rel(track[1], snaps[1][1]), TOL_S)
        _, h2 = self._extend(q, k, v, g, beta, cu, _pool(3), idx,
                             return_intermediate_states=True)
        self.assertEqual(tuple(h2.shape), (1, 3 + 2, H, K, K))

    def test_prefill_then_decode_then_verify(self):
        """Prefill a prompt with the SDA kernel, then decode tokens with SGLang's
        Triton recurrent kernel from the same pool row (capped logits): every
        output must follow the fp64 recurrence of the whole sequence."""
        P, D = 300, 6
        q, k, v, g, _, cu = _inputs([P + D], blogit=5.0)
        torch.manual_seed(5)
        blogit = 5.0 + torch.randn(1, P + D, H, device=DEV) * 2  # many above the cap
        capped = blogit.clamp(max=self.cap)
        beta = 2 * torch.sigmoid(capped)
        _, _, _ = None, None, None
        pool = torch.zeros(3, H, K, K, device=DEV)
        idx = torch.tensor([2], device=DEV, dtype=torch.int32)
        o_pre = self._extend(q[:, :P], k[:, :P], v[:, :P], g[:, :P], beta[:, :P],
                             torch.tensor([0, P], device=DEV, dtype=torch.int32), pool, idx)
        verify_pool = pool.clone()
        outs = [o_pre]
        for t in range(P, P + D):
            sl = slice(t, t + 1)
            outs.append(self._decode(q[:, sl], k[:, sl], v[:, sl], g[:, sl],
                                     capped[:, sl], pool, idx).view(1, 1, H, K))
        ro, rf, _ = _ref(q, k, v, g, beta, self.A_log, self.dt_bias,
                         torch.tensor([0, P + D], device=DEV, dtype=torch.int32))
        o = torch.cat(outs, 1)
        self.assertLess(_rel(o[:, P:], ro[:, P:]), TOL_O)
        self.assertLess(_rel(pool[2:3], rf), TOL_S)

        # target_verify: the D draft tokens in one call, state left untouched,
        # per-step states cached; outputs equal the step-by-step decode.
        from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
            fused_sigmoid_gating_delta_rule_update,
        )

        inter = torch.zeros(1, D, H, K, K, device=DEV)
        sl = slice(P, P + D)
        ov = fused_sigmoid_gating_delta_rule_update(
            A_log=self.A_log.view(1, 1, H, 1), dt_bias=self.dt_bias, q=q[:, sl], k=k[:, sl],
            v=v[:, sl], a=g[0, sl].reshape(D, H * K), b=capped[:, sl],
            initial_state_source=verify_pool, initial_state_indices=idx,
            cu_seqlens=torch.tensor([0, D], device=DEV, dtype=torch.int32),
            use_qk_l2norm_in_kernel=True, softplus_beta=1.0, softplus_threshold=20.0,
            is_kda=True, disable_state_update=True, intermediate_states_buffer=inter,
            intermediate_state_indices=torch.zeros(1, device=DEV, dtype=torch.int32),
            cache_steps=D, beta_scale=2.0)
        self.assertLess(_rel(ov.view(1, D, H, K), ro[:, P:]), TOL_O)
        self.assertLess(_rel(inter[0, -1], rf[0]), TOL_S)

    def test_mixed_batch(self):
        """A mixed batch (--enable-mixed-chunk): decode rows of running requests
        ride the chunked prefill as 1-token sequences next to a new prompt. They
        must match the decode kernel run on the same rows."""
        n_dec, P = 48, 513
        lens = [P] + [1] * n_dec
        q, k, v, g, beta, cu = _inputs(lens, blogit=3.0)
        blogit = torch.logit(beta / 2).clamp(max=self.cap)
        beta = 2 * torch.sigmoid(blogit)
        pool = _pool(n_dec + 2)
        pool_dec = pool.clone()
        idx = torch.arange(1, n_dec + 2, device=DEV, dtype=torch.int32)
        h0 = pool[idx.long()].clone()
        o = self._extend(q, k, v, g, beta, cu, pool, idx)
        ro, rf, _ = _ref(q, k, v, g, beta, self.A_log, self.dt_bias, cu, h0)
        self.assertLess(_rel(o, ro), TOL_O)
        self.assertLess(_rel(pool[idx.long()], rf), TOL_S)
        # The same decode rows through the decode kernel.
        sl = slice(P, P + n_dec)
        od = self._decode(q[:, sl], k[:, sl], v[:, sl], g[:, sl], blogit[:, sl],
                          pool_dec, idx[1:])
        self.assertLess(_rel(o[:, sl], od.view(1, n_dec, H, K).double()), TOL_O)
        self.assertLess(_rel(pool[idx.long()[1:]], pool_dec[idx.long()[1:]].double()), TOL_S)

    def test_non_grouped_gate_is_refused(self):
        q, k, v, g, beta, cu = _inputs([40])
        g = g.clone()
        g[..., 1] += 1.0  # breaks the 16-lane block of group 0
        with self.assertRaises(ValueError):
            self._extend(q, k, v, g, beta, cu, _pool(2),
                         torch.tensor([1], device=DEV, dtype=torch.int32))

    def test_kda_backend_routes_the_sda_layer(self):
        """KDAAttnBackend.forward_extend runs the layer's ``sda_prefill`` and never
        its KDA kernel dispatcher (stubbed to fail)."""
        from types import SimpleNamespace

        from sglang.srt.layers.attention.linear.kda_backend import KDAAttnBackend

        lens = [100, 1, 70]
        q, k, v, g, beta, cu = _inputs(lens)
        T = sum(lens)
        mixed_qkv = torch.cat([q[0].flatten(1), k[0].flatten(1), v[0].flatten(1)], -1)
        pool = _pool(4)
        idx = torch.tensor([1, 3, 2], device=DEV, dtype=torch.int32)
        h0 = pool[idx.long()].clone()

        class _NoDispatch:
            def extend(self, *a, **kw):
                raise AssertionError("the KDA kernel dispatcher ran for an SDA layer")

        be = KDAAttnBackend.__new__(KDAAttnBackend)
        be.prefill_graph_metadata = None
        be.accept_lens_pool = None
        be.kernel_dispatcher = _NoDispatch()
        be.forward_metadata = SimpleNamespace(
            query_start_loc=cu, mamba_cache_indices=idx, logical_num_tokens=None,
            has_mamba_track_mask=False, track_chunk_idx=None, track_ssm_h_src=None,
            track_ssm_h_batch_src=None, state_checkpoint_cu_starts=None,
            num_state_checkpoints=0, state_checkpoint_every_n_tokens=None,
            state_checkpoint_indices=None)
        cache = SimpleNamespace(conv=[torch.zeros(4, 3, 3 * H * K, device=DEV)], temporal=pool)
        be.req_to_token_pool = SimpleNamespace(mamba2_layer_cache=lambda layer_id: cache)
        be._convolve_prefill = lambda layer, fb, mixed, conv, ci: mixed  # conv tested elsewhere
        layer = SimpleNamespace(
            layer_id=0, q_dim=H * K, k_dim=H * K, v_dim=H * K, head_q_dim=K, head_k_dim=K,
            head_v_dim=K, A_log=self.A_log.view(1, 1, H, 1), dt_bias=self.dt_bias,
            lower_bound=None, beta_scale=2.0, sda_prefill=self.sda)
        fb = SimpleNamespace(
            forward_mode=SimpleNamespace(is_target_verify=lambda: False,
                                         is_draft_extend_v2=lambda: False),
            extend_prefix_lens=torch.zeros(3, device=DEV, dtype=torch.int32),
            extend_seq_lens_cpu=lens)
        o = be.forward_extend(layer, fb, mixed_qkv, g, beta)
        ro, rf, _ = _ref(q, k, v, g, beta, self.A_log, self.dt_bias, cu, h0)
        self.assertLess(_rel(o, ro), TOL_O)
        self.assertLess(_rel(pool[idx.long()], rf), TOL_S)
        self.assertEqual(o.shape[1], T)


@unittest.skipIf(SKIP, SKIP)
class TestSolarOpen3ModelWiring(unittest.TestCase):
    """The model side: on ROCm an SDA checkpoint's layers carry aiter's flash_sda
    prefill, an exact-erase checkpoint is then served (not refused), and the layer
    caps the beta logit for the decode / verify kernels."""

    @staticmethod
    def _config(**kw):
        from types import SimpleNamespace

        base = dict(linear_attn="sda", sda_decay_groups=G, exact_erase=True)
        base.update(kw)
        return SimpleNamespace(**base)

    def test_exact_erase_checkpoint_is_served(self):
        from sglang.srt.models.solar_open3 import SolarOpen3ForCausalLM

        model = SolarOpen3ForCausalLM.__new__(SolarOpen3ForCausalLM)
        object.__setattr__(model, "config", self._config())
        model._refuse_unserved_exact_erase()  # no FlashSDA needed on ROCm

    def test_exact_erase_refused_without_the_aiter_kernel(self):
        import sys

        from sglang.srt.models.solar_open3 import SolarOpen3ForCausalLM

        model = SolarOpen3ForCausalLM.__new__(SolarOpen3ForCausalLM)
        object.__setattr__(model, "config", self._config())
        name = "aiter.ops.triton._triton_kernels.chunk_delta_attn.flash_sda"
        saved = sys.modules.get(name)
        sys.modules[name] = None  # import fails: an aiter without flash_sda
        try:
            with self.assertRaises(RuntimeError):
                model._refuse_unserved_exact_erase()
        finally:
            if saved is None:
                del sys.modules[name]
            else:
                sys.modules[name] = saved

    def test_prefill_object(self):
        p = ADP.SdaRocmPrefill(groups=G, exact_erase=True)
        self.assertEqual((p.groups, p.exact_erase), (G, True))
        with self.assertRaises(TypeError):
            ADP.SdaRocmPrefill(groups=G, exact_erase=1)

    def test_layer_caps_the_beta_logit(self):
        from sglang.srt.models.solar_open3 import SolarOpen3KimiDeltaAttention

        layer = SolarOpen3KimiDeltaAttention.__new__(SolarOpen3KimiDeltaAttention)
        cap = ADP.beta_logit_cap(2.0)
        object.__setattr__(layer, "_sda_beta_logit_cap", cap)
        logit = torch.tensor([[-3.0, 4.0, 4.9, 12.0]], device=DEV, dtype=BF)
        qkv, beta, f, gg = layer._cap_beta_logit(("qkv", logit, "f", "g"))
        self.assertEqual((qkv, f, gg), ("qkv", "f", "g"))
        self.assertEqual(beta.dtype, torch.float32)
        act = 2 * torch.sigmoid(beta)
        self.assertTrue(torch.equal(beta[:, :2], logit[:, :2].float()))  # below: untouched
        self.assertLessEqual(float(act.max()), BETA_MAX + 1e-6)
        self.assertGreaterEqual(float(act[0, 3]), BETA_MAX)
        object.__setattr__(layer, "_sda_beta_logit_cap", None)
        self.assertIs(layer._cap_beta_logit(("q", logit, "f", "g"))[1], logit)


if __name__ == "__main__":
    unittest.main()
