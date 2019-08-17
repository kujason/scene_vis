import numpy as np

from core import transform_utils


class TransformUtilsTest:

    def test_np_get_tr_mat(self):

        ry_0 = 0.0
        t_0 = np.zeros(3)

        ry_90 = np.deg2rad(90.0)
        t_246 = [2.0, 4.0, 6.0]

        # Rotation: 0, translation: variable
        tr_mat_0_0 = transform_utils.np_get_tr_mat(ry_0, t_0)
        tr_mat_0_246 = transform_utils.np_get_tr_mat(ry_0, t_246)

        # Rotation: 90, translation: 0
        tr_mat_90_0 = transform_utils.np_get_tr_mat(ry_90, t_0)

        # Check for expected values
        np.testing.assert_allclose(tr_mat_0_0, np.eye(4))

        exp_0_246_mat = np.eye(4)
        exp_0_246_mat[0:3, 3] = [2.0, 4.0, 6.0]
        np.testing.assert_allclose(tr_mat_0_246, exp_0_246_mat)

        exp_90_0_mat = np.eye(4)
        exp_90_0_mat[0:3, 0:3] = [
            [+0.0, +0.0, +1.0],
            [+0.0, +1.0, +0.0],
            [-1.0, +0.0, +0.0],
        ]
        np.testing.assert_allclose(tr_mat_90_0, exp_90_0_mat, atol=1E-7)
