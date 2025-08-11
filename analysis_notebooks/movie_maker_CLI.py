import os
from pathlib import Path
import osh5vis
import osh5io
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import argparse
from asyncio import subprocess

# from ..src.make_movie_mp import movie, phase_space_movie 
class CLI_MovieMaker:
    def __init__(self, root_dir=None):
        self.root_dir = Path(root_dir)
        self.gyrotime = None
        self.dpi = 100

    def set_root_dir(self, path):
        self.root_dir = Path(path).resolve()
        if not self.root_dir.is_dir():
            print("Invalid directory.")
            return

    def set_gyrotime(self, gyrotime):
        self.gyrotime = gyrotime

    def set_dpi(self, dpi):
        self.dpi = dpi

    def tmp_vminmax(self, path_to_data, vlimits_override=None):
        '''
        Note that vmin and vmax would be different for each quantity that you wish to plot.

        '''
        if vlimits_override is not None:
            self.vmin = vlimits_override[0]
            self.vmax = vlimits_override[1]
            return self.vmin, self.vmax
        
        parent_dir = Path(path_to_data).resolve()
        frames = sorted(list(parent_dir.glob('*.h5')))
        if not frames:
            print("No .h5 files found in the directory.")
            raise ValueError("No .h5 files found in {}.".format(parent_dir))

        data_middle = osh5io.read_h5(str(frames[len(frames)//2]))
        vmin = np.min(data_middle)
        vmax = np.max(data_middle)
        self.vmin = vmin
        self.vmax = vmax
        return self.vmin, self.vmax

    def _clear_terminal(self):
        """Clear the terminal screen"""
        os.system('clear' if os.name == 'posix' else 'cls') 

    def _look_through_root(self):
        diagnostics = []
        for item in self.root_dir.rglob('*.h5'):
            parent_dir = item.parent.relative_to(self.root_dir)
            if parent_dir not in diagnostics:
                diagnostics.append(parent_dir)

        if not diagnostics:
            print("No subdirectories found.")
            return

        return diagnostics

    def frame_generator(self, path_to_frame):
        data = osh5io.read_h5(path_to_frame)

        if len(np.shape(data)) == 1:
            osh5vis.osplot(data, ylim=[self.vmin, self.vmax])
        else:
            osh5vis.osplot(data, vmax=self.vmax, vmin=self.vmin)

        if self.gyrotime:
            plt.title(f"{data.run_attrs['NAME']} {np.round(data.run_attrs['TIME'][0]/self.gyrotime, 2)}" + r" $\omega_{ci}$")

        plt.savefig(f'tmp/{path_to_frame.split("-")[-1].strip('.h5')}.png', dpi=self.dpi)

    def save_movie(self, path_to_tmp, fps=10, x=1920, y=1080, output_name="output"):
        path_to_tmp = Path(path_to_tmp).resolve()
        png_pattern = str(path_to_tmp / '*.png')
        output_file = str(path_to_tmp / f"../{output_name}.mp4")
        subprocess.call([
            "ffmpeg", "-framerate", str(fps), "-pattern_type", "glob", "-i", png_pattern,
            "-c:v", "libx264", "-vf", f"scale={x}:{y},format=yuv420p", output_file
        ])


if __name__ == "__main__":
    # Main execution block
    parser = argparse.ArgumentParser(description="Run simplified MagShockZ analysis and generate OSIRIS input file.")
    parser.add_argument('-d', '--data_path', type=str, default=None, help="Path to OSIRIS MS directory")
    parser.add_argument('-g', '--gyrotime', type=float, default=None, help="Gyrofrequency time in OSIRIS units")
    cli = CLI_MovieMaker(root_dir=parser.parse_args().data_path)
    cli.set_gyrotime(parser.parse_args().gyrotime)
    if cli.root_dir is None:
        data_path = input("Enter the path to the OSIRIS MS directory: ")
        cli.set_root_dir(Path(data_path).resolve())
    while True:
        cli._clear_terminal()
        subdirs = cli._look_through_root()
        if not subdirs:
            print("No subdirectories found.")
            break
        for i in range(len(subdirs)):
            print(f"{i}:    {subdirs[i]}")
        
        user_input = input("\nEnter the number of the subdirectory to animate (or 'exit' to quit): ").strip()
        
        if user_input.lower() == 'exit':
            print("Exiting...")
            break
        
        try:
            choice = int(user_input)
            if 0 <= choice < len(subdirs):
                print(f"You selected: {subdirs[choice]}")
                print("full path to data:", cli.root_dir / subdirs[choice])

                choice_dir = Path(cli.root_dir / subdirs[choice]).resolve()
                frames = sorted(list(choice_dir.glob('*.h5')))
                n_jobs = cpu_count()
                
                cli.tmp_vminmax(choice_dir)
                print(f"vmin: {cli.vmin}, vmax: {cli.vmax} \n")
                vlimits_override = input("Enter vmin and vmax separated by a space (or press Enter to continue): ").strip()
                if vlimits_override:
                    try:
                        vlimits_override = list(map(float, vlimits_override.split()))
                        cli.tmp_vminmax(choice_dir, vlimits_override=vlimits_override)
                    except ValueError:
                        print("Invalid input. Using default vmin and vmax.")
                if Path('tmp').exists():
                    print("Removing old 'tmp' directory.")
                    for item in Path('tmp').glob('*'):
                        if item.is_file():
                            item.unlink()
                        elif item.is_dir():
                            item.rmdir()
                    Path('tmp').rmdir()
                Path('tmp').mkdir()
                with Pool(n_jobs) as pool:
                    results = pool.map(cli.frame_generator, [str(frame) for frame in frames])

                print("Frames generated and saved as PNG files in 'tmp' directory.")

                cli.save_movie(scan_dir = "tmp", fname = '')

            else:
                print(f"Invalid selection. Please enter a number between 0 and {len(subdirs)-1}.")
                from time import sleep
                sleep(2)
        except ValueError:
            print("Please enter a valid number or 'exit' to quit.")

