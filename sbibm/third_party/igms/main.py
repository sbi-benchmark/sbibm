"""
Code by Dougal J. Sutherland:
- https://github.com/dougalsutherland/igms
- https://github.com/dougalsutherland/mlss-testing
- https://github.com/dougalsutherland/ds3-kernels/


                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
from functools import wraps

import numpy as np
import torch


################################################################################
# Utils
def as_tensors(X, *rest):
    "Calls as_tensor on a bunch of args, all of the first's device and dtype."
    X = torch.as_tensor(X)
    return [X] + [
        None if r is None else torch.as_tensor(r, device=X.device, dtype=X.dtype)
        for r in rest
    ]


def maybe_squeeze(X, dim):
    "Like torch.squeeze, but don't crash if dim already doesn't exist."
    return torch.squeeze(X, dim) if dim < len(X.shape) else X


################################################################################


def _cache(f):
    # Only works when the function takes no or simple arguments!
    @wraps(f)
    def wrapper(self, *args):
        key = (f.__name__,) + tuple(args)
        if key in self._cache:
            return self._cache[key]
        self._cache[key] = val = f(self, *args)
        return val

    return wrapper


################################################################################
# Kernel base class

_name_map = {"X": 0, "Y": 1, "Z": 2}


class LazyKernel(torch.nn.Module):
    """
    Base class that allows computing kernel matrices among a bunch of datasets,
    only computing the matrices when we use them.

    Constructor arguments:
        - A bunch of matrices we'll compute the kernel among.
          2d tensors, with second dimension agreeing, or None;
          None is a special value meaning to use the first entry X.
          (This is more efficient than passing the same tensor again.)

    Access the results with:
      - K[0, 1] to get the Tensor between parts 0 and 1.
      - K.XX, K.XY, K.ZY, etc: shortcuts, with X=0, Y=1, Z=2.
      - K.matrix(0, 1) or K.XY_m: returns a Matrix subclass (see below).
    """

    def __init__(self, X, *rest):
        super().__init__()
        self._cache = {}
        if not hasattr(self, "const_diagonal"):
            self.const_diagonal = False

        # want to use pytorch buffer for parts
        # but can't assign a list to those, so munge some names
        X, *rest = as_tensors(X, *rest)
        if len(X.shape) < 2:
            raise ValueError(
                "LazyKernel expects parameters to be at least 2d. "
                "If your data is 1d, make it [n, 1] with X[:, np.newaxis]."
            )

        self.register_buffer("_part_0", X)
        self.n_parts = 1
        for p in rest:
            self.append_part(p)

    @property
    def X(self):
        return self._part_0

    def _part(self, i):
        return self._buffers[f"_part_{i}"]

    def part(self, i):
        p = self._part(i)
        return self.X if p is None else p

    def n(self, i):
        return self.part(i).shape[0]

    @property
    def ns(self):
        return [self.n(i) for i in range(self.n_parts)]

    @property
    def parts(self):
        return [self.part(i) for i in range(self.n_parts)]

    @property
    def dtype(self):
        return self.X.dtype

    @property
    def device(self):
        return self.X.device

    def __repr__(self):
        return f"<{type(self).__name__}({', '.join(str(n) for n in self.ns)})>"

    def _compute(self, A, B):
        """
        Compute the kernel matrix between A and B.

        Might get called with A = X, B = X, or A = X, B = Y, etc.

        Should return a tensor of shape [A.shape[0], B.shape[0]].

        This default, slow, version calls self._compute_one(a, b) in a loop.
        If you override this, you don't need to implement _compute_one at all.

        If you implement _precompute, this gets added to the signature here:
            self._compute(A, *self._precompute(A), B, *self._precompute(B)).
        The default _precompute returns an empty tuple, so it's _compute(A, B),
        but if you make a _precompute that returns [A_squared, A_cubed] then it's
            self._compute(A, A_squared, A_cubed, B, B_squared, B_cubed).
        """
        return torch.stack(
            [
                torch.stack([torch.as_tensor(self._compute_one(a, b)) for b in B])
                for a in A
            ]
        )

    def _compute_one(self, a, b):
        raise NotImplementedError(
            f"{type(self).__name__}: need to implement _compute or _compute_one"
        )

    def _precompute(self, A):
        """
        Compute something extra for each part A.

        Can be used to share computation between kernel(X, X) and kernel(X, Y).

        We end up calling basically (but with caching)
            self._compute(A, *self._precompute(A), B, *self._precompute(B))
        This default _precompute returns an empty tuple, so it's
            self._compute(A, B)
        But if you return [A_squared], it'd be
            self._compute(A, A_squared, B, B_squared)
        and so on.
        """
        return ()

    @_cache
    def _precompute_i(self, i):
        p = self._part(i)
        if p is None:
            return self._precompute_i(0)
        return self._precompute(p)

    @_cache
    def __getitem__(self, k):
        try:
            i, j = k
        except ValueError:
            raise KeyError("You should index kernels with pairs")

        A = self._part(i)
        if A is None:
            return self[0, j]

        B = self._part(j)
        if B is None:
            return self[i, 0]

        if i > j:
            return self[j, i].t()

        A_info = self._precompute_i(i)
        B_info = self._precompute_i(j)
        return self._compute(A, *A_info, B, *B_info)

    @_cache
    def matrix(self, i, j):
        if self._part(i) is None:
            return self.matrix(0, j)

        if self._part(j) is None:
            return self.matrix(i, 0)

        k = self[i, j]
        if i == j:
            return as_matrix(k, const_diagonal=self.const_diagonal, symmetric=True)
        else:
            return as_matrix(k)

    @_cache
    def joint(self, *inds):
        if not inds:
            return self.joint(*range(self.n_parts))
        return torch.cat([torch.cat([self[i, j] for j in inds], 1) for i in inds], 0)

    @_cache
    def joint_m(self, *inds):
        if not inds:
            return self.joint_m(*range(self.n_parts))
        return as_matrix(
            self.joint(*inds), const_diagonal=self.const_diagonal, symmetric=True
        )

    def __getattr__(self, name):
        # self.X, self.Y, self.Z
        if name in _name_map:
            i = _name_map[name]
            if i < self.n_parts:
                return self.part(i)
            else:
                raise AttributeError(f"have {self.n_parts} parts, asked for {i}")

        # self.XX, self.XY, self.YZ, etc; also self.XX_m
        ret_matrix = False
        if len(name) == 4 and name.endswith("_m"):
            ret_matrix = True
            name = name[:2]

        if len(name) == 2:
            i = _name_map.get(name[0], np.inf)
            j = _name_map.get(name[1], np.inf)
            if i < self.n_parts and j < self.n_parts:
                return self.matrix(i, j) if ret_matrix else self[i, j]
            else:
                raise AttributeError(f"have {self.n_parts} parts, asked for {i}, {j}")

        return super().__getattr__(name)

    def _invalidate_cache(self, i):
        for k in list(self._cache.keys()):
            if (
                i in k[1:]
                or any(isinstance(arg, tuple) and i in arg for arg in k[1:])
                or k in [("joint",), ("joint_m",)]
            ):
                del self._cache[k]

    def drop_last_part(self):
        assert self.n_parts >= 2
        i = self.n_parts - 1
        self._invalidate_cache(i)
        del self._buffers[f"_part_{i}"]
        self.n_parts -= 1

    def change_part(self, i, new):
        assert i < self.n_parts
        if new is not None and new.shape[1:] != self.X.shape[1:]:
            raise ValueError(f"X has shape {self.X.shape}, new entry has {new.shape}")
        self._invalidate_cache(i)
        self._buffers[f"_part_{i}"] = new

    def append_part(self, new):
        if new is not None and new.shape[1:] != self.X.shape[1:]:
            raise ValueError(f"X has shape {self.X.shape}, new entry has {new.shape}")
        self._buffers[f"_part_{self.n_parts}"] = new
        self.n_parts += 1

    def __copy__(self):
        """
        Doesn't deep-copy the data tensors, but copies dictionaries so that
        change_part/etc don't affect the original.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        to_copy = {"_cache", "_buffers", "_parameters", "_modules"}
        result.__dict__.update(
            {k: v.copy() if k in to_copy else v for k, v in self.__dict__.items()}
        )
        return result

    def _apply(self, fn):  # used in to(), cuda(), etc
        super()._apply(fn)
        for key, val in self._cache.items():
            if val is not None:
                self._cache[key] = fn(val)
        return self

    def as_tensors(self, *args, **kwargs):
        "Helper that makes everything a tensor with self.X's type."
        kwargs.setdefault("device", self.X.device)
        kwargs.setdefault("dtype", self.X.dtype)
        return tuple(None if r is None else torch.as_tensor(r, **kwargs) for r in args)


################################################################################
# Matrix wrappers that cache sums / etc. Including various subclasses; see
# as_matrix() to pick between them appropriately.

# TODO: could support a matrix transpose that shares the cache appropriately


class Matrix:
    def __init__(self, M, const_diagonal=False):
        self.mat = M = torch.as_tensor(M)
        self.m, self.n = self.shape = M.shape
        self._cache = {}

    @_cache
    def row_sums(self):
        return self.mat.sum(0)

    @_cache
    def col_sums(self):
        return self.mat.sum(1)

    @_cache
    def row_sums_sq_sum(self):
        sums = self.row_sums()
        return sums @ sums

    @_cache
    def col_sums_sq_sum(self):
        sums = self.col_sums()
        return sums @ sums

    @_cache
    def sum(self):
        if "row_sums" in self._cache:
            return self.row_sums().sum()
        elif "col_sums" in self._cache:
            return self.col_sums().sum()
        else:
            return self.mat.sum()

    def mean(self):
        return self.sum() / (self.m * self.n)

    @_cache
    def sq_sum(self):
        flat = self.mat.view(-1)
        return flat @ flat

    def __repr__(self):
        return f"<{type(self).__name__}, {self.m} by {self.n}>"


class SquareMatrix(Matrix):
    def __init__(self, M):
        super().__init__(M)
        assert self.m == self.n

    @_cache
    def diagonal(self):
        return self.mat.diagonal()

    @_cache
    def trace(self):
        return self.mat.trace()

    @_cache
    def sq_trace(self):
        diag = self.diagonal()
        return diag @ diag

    @_cache
    def offdiag_row_sums(self):
        return self.row_sums() - self.diagonal()

    @_cache
    def offdiag_col_sums(self):
        return self.col_sums() - self.diagonal()

    @_cache
    def offdiag_row_sums_sq_sum(self):
        sums = self.offdiag_row_sums()
        return sums @ sums

    @_cache
    def offdiag_col_sums_sq_sum(self):
        sums = self.offdiag_col_sums()
        return sums @ sums

    @_cache
    def offdiag_sum(self):
        return self.offdiag_row_sums().sum()

    def offdiag_mean(self):
        return self.offdiag_sum() / (self.n * (self.n - 1))

    @_cache
    def offdiag_sq_sum(self):
        return self.sq_sum() - self.sq_trace()


class SymmetricMatrix(SquareMatrix):
    def col_sums(self):
        return self.row_sums()

    def sums(self):
        return self.row_sums()

    def offdiag_col_sums(self):
        return self.offdiag_row_sums()

    def offdiag_sums(self):
        return self.offdiag_row_sums()

    def col_sums_sq_sum(self):
        return self.row_sums_sq_sum()

    def sums_sq_sum(self):
        return self.row_sums_sq_sum()

    def offdiag_col_sums_sq_sum(self):
        return self.offdiag_row_sums_sq_sum()

    def offdiag_sums_sq_sum(self):
        return self.offdiag_row_sums_sq_sum()


class ConstDiagMatrix(SquareMatrix):
    def __init__(self, M, diag_value):
        super().__init__(M)
        self.diag_value = diag_value

    @_cache
    def diagonal(self):
        return self.mat.new_full((1,), self.diag_value)

    def trace(self):
        return self.n * self.diag_value

    def sq_trace(self):
        return self.n * (self.diag_value ** 2)


class SymmetricConstDiagMatrix(ConstDiagMatrix, SymmetricMatrix):
    pass


def as_matrix(M, const_diagonal=False, symmetric=False):
    if symmetric:
        if const_diagonal is not False:
            return SymmetricConstDiagMatrix(M, diag_value=const_diagonal)
        else:
            return SymmetricMatrix(M)
    elif const_diagonal is not False:
        return ConstDiagMatrix(M, diag_value=const_diagonal)
    elif M.shape[0] == M.shape[1]:
        return SquareMatrix(M)
    else:
        return Matrix(M)


################################################################################


def mmd2_u_stat_variance(K, inds=(0, 1)):
    """
    Estimate MMD variance with estimator from https://arxiv.org/abs/1906.02104.

    K should be a LazyKernel; we'll compare the parts in inds,
    default (0, 1) to use K.XX, K.XY, K.YY.
    """
    i, j = inds

    m = K.n(i)
    assert K.n(j) == m

    XX = K.matrix(i, i)
    XY = K.matrix(i, j)
    YY = K.matrix(j, j)

    mm = m * m
    mmm = mm * m
    m1 = m - 1
    m1_m1 = m1 * m1
    m1_m1_m1 = m1_m1 * m1
    m2 = m - 2
    mdown2 = m * m1
    mdown3 = mdown2 * m2
    mdown4 = mdown3 * (m - 3)
    twom3 = 2 * m - 3

    return (
        (4 / mdown4) * (XX.offdiag_sums_sq_sum() + YY.offdiag_sums_sq_sum())
        + (4 * (mm - m - 1) / (mmm * m1_m1))
        * (XY.row_sums_sq_sum() + XY.col_sums_sq_sum())
        - (8 / (mm * (mm - 3 * m + 2)))
        * (XX.offdiag_sums() @ XY.col_sums() + YY.offdiag_sums() @ XY.row_sums())
        + 8 / (mm * mdown3) * ((XX.offdiag_sum() + YY.offdiag_sum()) * XY.sum())
        - (2 * twom3 / (mdown2 * mdown4)) * (XX.offdiag_sum() + YY.offdiag_sum())
        - (4 * twom3 / (mmm * m1_m1_m1)) * XY.sum() ** 2
        - (2 / (m * (mmm - 6 * mm + 11 * m - 6)))
        * (XX.offdiag_sq_sum() + YY.offdiag_sq_sum())
        + (4 * m2 / (mm * m1_m1_m1)) * XY.sq_sum()
    )


################################################################################


class ExpQuadKernel(LazyKernel):
    def __init__(self, *parts, sigma=1):
        super().__init__(*parts)
        self.sigma = sigma
        self.const_diagonal = 1  # Says that k(x, x) = 1 for any x.
        # Just a slight optimization; not really necessary.

    # TODO: implement _compute (maybe with _precompute) or _compute_one
    def _precompute(self, A):
        # Squared norms of each data point
        return [torch.einsum("ij,ij->i", A, A)]

    def _compute(self, A, A_sqnorms, B, B_sqnorms):
        D2 = A_sqnorms[:, None] + B_sqnorms[None, :] - 2 * (A @ B.t())
        return torch.exp(D2 / (-2 * self.sigma ** 2))


def mean_difference(X, Y, squared=False):
    X, Y = [maybe_squeeze(t, 1) for t in as_tensors(X, Y)]
    assert len(X.shape) == len(Y.shape) == 1

    # TODO: compute mean difference of X and Y in `result`
    result = X.mean() - Y.mean()

    return (result * result) if squared else result


def median_distance(Z):
    return torch.median(torch.pdist(Z))


def mmd2_biased(K):
    return K.XX_m.mean() + K.YY_m.mean() - 2 * K.XY_m.mean()


def mmd2_unbiased(K):
    return K.XX_m.offdiag_mean() + K.YY_m.offdiag_mean() - 2 * K.XY_m.mean()


def mmd2_u_stat(K):
    assert K.ns[0] == K.ns[1]
    return K.XX_m.offdiag_mean() + K.YY_m.offdiag_mean() - 2 * K.XY_m.offdiag_mean()
