import numpy as np


# All credit/blame to Justin Gray.
def brentv(f, x0, x1, max_iter=50, tolerance=1e-5):

    x0 = np.atleast_1d(x0)
    x1 = np.atleast_1d(x1)

    shape = x0.shape

    fx0 = f(x0)
    fx1 = f(x1)

    assert np.any((fx0 * fx1) <= 0), "Root not bracketed"

    # need to flip bounds?
    to_flip = np.where(np.abs(fx0) < np.abs(fx1))
    x0[to_flip], x1[to_flip] = x1[to_flip], x0[to_flip]
    fx0[to_flip], fx1[to_flip] = fx1[to_flip], fx0[to_flip]

    x2, fx2 = x0.copy(), fx0.copy()

    mflag = np.ones(shape, dtype=bool)
    not_mflag = np.invert(mflag)

    # mflag = True

    def mask_vecs(mask):
        x0_m = x0[mask]
        x1_m = x1[mask]
        x2_m = x2[mask]
        fx0_m = fx0[mask]
        fx1_m = fx1[mask]
        fx2_m = fx2[mask]

        return x0_m, x1_m, x2_m, fx0_m, fx1_m, fx2_m

    d = x2.copy()

    steps_taken = 0
    success = np.all(np.abs(x1 - x0) < tolerance)
    while steps_taken < max_iter and not success:

        fx0 = f(x0)
        fx1 = f(x1)
        fx2 = f(x2)

        new = np.empty(shape)
        mask = np.logical_and(fx0 != fx2, fx1 != fx2)
        x0_m, x1_m, x2_m, fx0_m, fx1_m, fx2_m = mask_vecs(mask)

        L0 = (x0_m * fx1_m * fx2_m) / ((fx0_m - fx1_m) * (fx0_m - fx2_m))
        L1 = (x1_m * fx0_m * fx2_m) / ((fx1_m - fx0_m) * (fx1_m - fx2_m))
        L2 = (x2_m * fx1_m * fx0_m) / ((fx2_m - fx0_m) * (fx2_m - fx1_m))
        new[mask] = L0 + L1 + L2

        not_mask = np.invert(mask)
        x0_m, x1_m, x2_m, fx0_m, fx1_m, fx2_m = mask_vecs(not_mask)

        new[not_mask] = x1_m - ((fx1_m * (x1_m - x0_m)) / (fx1_m - fx0_m))

        check_0 = np.logical_or(new < ((3 * x0 + x1) / 4), new > x1)
        check_1 = np.logical_and(mflag, abs(new - x1) >= abs(x1 - x2) / 2)
        check_2 = np.logical_and(not_mflag, np.any(abs(new - x1) >= abs(x2 - d) / 2))
        check_3 = np.logical_and(mflag, abs(x1 - x2) < tolerance)
        check_4 = np.logical_and(not_mflag, abs(x2 - d) < tolerance)

        mask = np.logical_or.reduce((check_0, check_1, check_2, check_3, check_4))
        x0_m, x1_m, x2_m, fx0_m, fx1_m, fx2_m = mask_vecs(mask)
        new[mask] = (x0_m + x1_m) / 2

        mflag[mask] = True
        not_mflag[mask] = False

        fnew = f(new)
        d, x2 = x2, x1.copy()

        # move the brackets
        move_right = np.where(fx0*fnew < 0)
        x1[move_right] = new[move_right]

        move_left = np.where(fx1*fnew < 0)
        x0[move_left] = new[move_left]

        # flip any reversed bounds
        to_flip = np.where(np.abs(fx0) < np.abs(fx1))
        x0[to_flip], x1[to_flip] = x1[to_flip], x0[to_flip]

        steps_taken += 1
        success = np.all(np.abs(x1 - x0) < tolerance)

    return x1, steps_taken, success


if __name__ == "__main__":
    N_POINTS = 3

    offset = np.linspace(10, 20, N_POINTS)

    def f(x):
        return (x-offset)**2-20

    left = offset.copy()
    right = 2*offset.copy()

    print(f(left))
    print(f(right))
    print()

    root, steps, success = brentv(f, left, right, tolerance=10e-5)
    print("root is:", root)
    print("steps taken:", steps)
