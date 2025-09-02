import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src.vysxd_analysis import get_osiris_quantity_2d

class NumericalInstabilityChecker:
    def __init__(self, ms_dir):
        self.ms_dir = ms_dir

    def get_div_E(self):
        try:
            e1, _, _, _, _ = get_osiris_quantity_2d(str(self.ms_dir / 'FLD/e1/'))
            e2, _, _, _, _ = get_osiris_quantity_2d(str(self.ms_dir / 'FLD/e2/'))
            e3, _, _, _, _ = get_osiris_quantity_2d(str(self.ms_dir / 'FLD/e3/'))
            self.div_E = np.gradient(e1, axis=0) + np.gradient(e2, axis=1) + np.gradient(e3, axis=2)
            return self.div_E
        except Exception as e:
            print('Failed to compute div(E):', e)
            return None
        
    def get_rho_tot(self):
        try:
            edens, _, _, _, _ = get_osiris_quantity_2d(str(self.ms_dir / 'DENSITY/electrons/charge/'))
            aldens, _, _, _, _ = get_osiris_quantity_2d(str(self.ms_dir / 'DENSITY/aluminum/charge/'))
            sidens, _, _, _, _ = get_osiris_quantity_2d(str(self.ms_dir / 'DENSITY/silicon/charge/'))

            self.rho_tot = edens + aldens + sidens
            return self.rho_tot
        except Exception as e:
            print('Failed to compute total charge density:', e)
            return None
    
    def get_curl_B(self):
        try:
            b1, _, _, _, _ = get_osiris_quantity_2d(str(self.ms_dir / 'FLD/b1/'))
            b2, _, _, _, _ = get_osiris_quantity_2d(str(self.ms_dir / 'FLD/b2/'))
            b3, _, _, _, _ = get_osiris_quantity_2d(str(self.ms_dir / 'FLD/b3/'))
            self.curl_B = np.gradient(b3, axis=1) - np.gradient(b2, axis=2), np.gradient(b1, axis=2) - np.gradient(b3, axis=0), np.gradient(b2, axis=0) - np.gradient(b1, axis=1)
            return self.curl_B
        except Exception as e:
            print('Failed to compute curl(B):', e)
            return None

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run numerical instability tests on OSIRIS MS directory.")
    parser.add_argument('-d', '--data_path', type=str, required=True, help="Path to OSIRIS MS directory")
    # args = parser.parse_args()
    ms_dir = 
    ms_dir = Path(args.data_path).resolve()
    if not ms_dir.is_dir():
        print("Invalid directory.")
        return
    print(f"Running tests on {ms_dir}")
    checker = NumericalInstabilityChecker(ms_dir)
    div_E = checker.get_div_E()
    rho_tot = checker.get_rho_tot()
    curl_B = checker.get_curl_B()
    print("All tests complete.")

if __name__ == "__main__":
    main()
