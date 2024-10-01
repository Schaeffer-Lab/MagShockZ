def alfven(B, ne, ion_mass):
    import numpy as np
    m_e = 9.10938356e-28
    c = 2.99792458e10
    alfven = B/np.sqrt(4*np.pi*ne*ion_mass*m_e)
    print(alfven/c)

alfven(150000,5e18,400)
