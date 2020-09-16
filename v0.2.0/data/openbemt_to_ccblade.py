import sys
import numpy as np


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('xtfile')
    parser.add_argument('ytfile')
    parser.add_argument('outfilepattern')
    args = parser.parse_args()

    if args.plot:
        import matplotlib.pyplot as plt

    with open(args.xtfile, mode='r') as f:
        comment, num_alpha, num_Re = f.readline().split()
        num_alpha = int(num_alpha)
        num_Re = int(num_Re)
        xt = np.loadtxt(f, delimiter=' ')

    xt.shape = (num_alpha, num_Re, -1)

    yt = np.loadtxt(args.ytfile, delimiter=' ')
    yt.shape = (num_alpha, num_Re, -1)

    if args.plot:
        fig, (ax_cl, ax_cd) = plt.subplots(nrows=2, sharex=True)

    for i in range(num_Re):
        # Check that the Reynolds numbers for this index are all the same.
        if np.std(xt[:, i, 1]) > 1e-10:
            print(f"warning: Reynolds numbers for index i = {i} appear to differ", file=sys.stderr)

        alpha = np.degrees(xt[:, i, 0])
        Re = xt[0, i, 1]

        cl = yt[:, i, 0]
        cd = yt[:, i, 1]

        cl_over_cd = cl/cd
        max_cl_over_cd = np.max(cl_over_cd)
        max_cl_over_cd_alpha = alpha[np.argmax(cl_over_cd)]

        fname = args.outfilepattern.format(i)
        header = f"Re = {Re}e6, max(Cl/Cd) = {max_cl_over_cd} at alpha = {max_cl_over_cd_alpha} deg"
        print(f"fname = {fname}, {header}")

        data = np.concatenate(
            [
                alpha[:, np.newaxis], cl[:, np.newaxis], cd[:, np.newaxis]
            ],
            axis=1)
        np.savetxt(fname, data, delimiter=' ', header=header)

        if args.plot:
            ax_cl.plot(alpha, cl, label=f'Re = {Re}e6')
            ax_cd.plot(alpha, cd, label=f'Re = {Re}e6')

    if args.plot:
        ax_cl.set_xlabel('alpha, deg')
        ax_cd.set_xlabel('alpha, deg')

        ax_cl.set_ylabel('Lift Coefficient')
        ax_cd.set_ylabel('Drag Coefficient')

        ax_cl.legend()
        ax_cd.legend()

        plt.show()
