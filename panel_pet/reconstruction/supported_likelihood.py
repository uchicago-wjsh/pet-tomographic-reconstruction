"""List-mode likelihood with explicit handling of zero-response events."""
import torch
import pytomography
from pytomography.likelihoods import PoissonLogLikelihood


class SupportedEventsPoissonLogLikelihood(PoissonLogLikelihood):
    """Exclude events whose forward response is zero on the active support.

    With parallelproj 1.10.2, real NEMA LORs with zero forward response can
    still backproject small values at a support boundary. The usual 1/(FP+delta)
    turns that leakage into a huge update. Zero-response observations have no
    usable row in the reconstruction model and must receive zero weight.
    Exclusion is permanent and counts are retained for the run's provenance.
    This implementation is for full-data MLEM, without additive backgrounds.
    """

    def __init__(self, system_matrix):
        super().__init__(system_matrix)
        self.supported_events = None
        self.exclusion_history = []

    def compute_gradient(self, object, subset_idx=None,
                         norm_BP_subset_method='subset_specific'):
        if subset_idx is not None or self.exists_additive_term:
            raise ValueError('Supported-event likelihood requires full-data MLEM without an additive term')
        fp = self.system_matrix.forward(object)
        if not torch.isfinite(fp).all() or (fp < 0).any():
            raise FloatingPointError('Forward projection contains negative or non-finite values')
        positive = fp > 0
        if self.supported_events is None:
            self.supported_events = positive
        else:
            self.supported_events &= positive
        excluded = int((~self.supported_events).sum().item())
        self.exclusion_history.append(excluded)
        iteration = len(self.exclusion_history)
        if iteration == 1 or excluded != self.exclusion_history[-2]:
            print(f'Event support at iteration {iteration}: excluded {excluded:,} '
                  f'of {fp.numel():,} zero-response events '
                  f'({100*excluded/fp.numel():.6f}%)', flush=True)
        self.projections_predicted = fp
        ratio = torch.where(self.supported_events,
                            torch.reciprocal(fp + pytomography.delta), 0.)
        return self.system_matrix.backward(ratio) - self._get_normBP(None)
