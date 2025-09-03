import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src.vysxd_analysis import get_osiris_quantity_2d

class NumericalInstabilityChecker:
    def __init__(self, ms_dir):
        self.ms_dir = Path(ms_dir)
        if not self.ms_dir.is_dir():
            print("Invalid directory.")
            return
        self.diags = self._look_through_MS()
        print("Found diagnostics:")
        for key, val in self.diags.items():
            print(f"key: {key}, val: {val}")

    def _look_through_MS(self):
            diagnostics = {} 
            for item in self.ms_dir.rglob('*.h5'):
                parent_dir = item.parent.relative_to(self.ms_dir)
                if parent_dir not in diagnostics:
                    diagnostics[str(parent_dir)] = item.parent.as_posix() + '/'

            if not diagnostics:
                print("No subdirectories found.")
                return

            return diagnostics

    def get_div_E(self):

        e1, __, dx, dy, _, _, _ = get_osiris_quantity_2d(self.diags['FLD/e1'])
        e2,  _,  _,  _, _, _, _ = get_osiris_quantity_2d(self.diags['FLD/e2'])
        e3,  _,  _,  _, _, _, _ = get_osiris_quantity_2d(self.diags['FLD/e3'])
        self.div_E = np.gradient(e1, dx, axis = 1) + np.gradient(e2, dy, axis = 2) # + 0  2D simulation, no variation in z
        return self.div_E
        
    def get_rho_tot(self):
        edens, _,  _,  _, _, _, _ = get_osiris_quantity_2d(self.diags['DENSITY/electrons/charge'])
        aldens, _,  _,  _, _, _, _ = get_osiris_quantity_2d(self.diags['DENSITY/aluminum/charge'])
        sidens, _,  _,  _, _, _, _ = get_osiris_quantity_2d(self.diags['DENSITY/silicon/charge'])

        self.edens = edens
        self.aldens = aldens
        self.sidens = sidens
        self.rho_tot = edens + aldens + sidens
        return self.rho_tot
    
    def get_curl_B(self):
        b1, dt, dx, dy, t, X, Y = get_osiris_quantity_2d(str(self.ms_dir / 'FLD/b1/'))
        b2,  _,  _,  _, _, _, _ = get_osiris_quantity_2d(str(self.ms_dir / 'FLD/b2/'))
        b3,  _,  _,  _, _, _, _ = get_osiris_quantity_2d(str(self.ms_dir / 'FLD/b3/'))
        self.curl_B = [np.gradient(b3, dy, axis = 2),
                       np.gradient(-b3, dx, axis=1),
                       np.gradient(b2, dx, axis=1) - np.gradient(b1, dy, axis=2)]

    def _frame_generator(self, path_to_frame):
        plt.clf()
        data = osh5io.read_h5(path_to_frame)

        if len(np.shape(data)) == 1:
            osh5vis.osplot(data[self.spatial_bounds[0]:self.spatial_bounds[1]], ylim=[self.vmin, self.vmax])
        else:
            osh5vis.osplot(data[self.spatial_bounds[0]:self.spatial_bounds[1], self.spatial_bounds[2]:self.spatial_bounds[3]], vmax=self.vmax, vmin=self.vmin)

        if self.gyrotime:
            plt.title(f"{data.run_attrs['NAME']} {np.round(data.run_attrs['TIME'][0]/self.gyrotime, 2)}" + r" $\omega_{ci}$")

        plt.savefig(f'tmp/{path_to_frame.split("-")[-1]}.png', dpi=self.dpi)

    def _save_movie(self, path_to_tmp, fps=40, x=1920, y=1080, output_name="output"):
        path_to_tmp = Path(path_to_tmp).resolve()
        png_pattern = str(path_to_tmp / '*.h5.png')
        output_file = str(path_to_tmp / f"../{output_name}.mp4")
        # I don't really understand how this works, but it really works
        subprocess.call([
            "ffmpeg", "-framerate", str(fps), "-pattern_type", "glob", "-i", png_pattern,
            "-c:v", "libx264", "-vf", f"scale={x}:{y},format=yuv420p", output_file
        ])



def main(ms_dir = "/mnt/cellar/dschneidinger/MagShockZ_pic/magshockz-v2.9.2d/MS"):
    print(f"Running tests on {ms_dir}")
    checker = NumericalInstabilityChecker(ms_dir)
    div_E = checker.get_div_E()
    rho_tot = checker.get_rho_tot()
    curl_B = checker.get_curl_B()
    print("All tests complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run numerical instability tests on OSIRIS MS directory.")
    parser.add_argument('-d', '--data_path', type=str, required=True, help="Path to OSIRIS MS directory")
    # args = parser.parse_args()
    main()
