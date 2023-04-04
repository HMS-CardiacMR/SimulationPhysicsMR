import numpy as np
import matplotlib.pylab as plt
import sycomore
from sycomore import units

def get_inhomogeneity(B0=1.5, inhomogeneity_ppm=0.0):
    gyromagnetic_ratio = 42.58e6  # Hz/T
    freq_rot = gyromagnetic_ratio * B0
    freq = gyromagnetic_ratio * B0 * (1 - inhomogeneity_ppm)
    delta_freq = (freq - freq_rot) * 1e-6
    return delta_freq * units.Hz

def get_species(T1, T2, delta_omega=0 * units.Hz):   
    return sycomore.Species(T1, T2, delta_omega=delta_omega)

def update_and_record_magnetization(M, record, t, update=None):
    if update is not None:
        M = update @ M
    record.append([t.convert_to(units.ms), M[:3] / M[3]])
    return M

def run_simulation_single_tissue(T1_ms, T2_ms, flip_angle_deg, step_size_ms, n_species, n_steps, step_refocus):
    
    delta_omega = get_inhomogeneity(B0=1.5, inhomogeneity_ppm=-0.2)
    T1 = T1_ms * units.ms
    T2 = T2_ms * units.ms
    n_species = int(n_species)

    species_baseline = [get_species(T1, T2, delta_omega=-delta_omega)]
    species_ensemble = species_baseline + [get_species(T1, T2, delta_omega=delta_omega*np.random.randn()) for _ in range(n_species-1)]

    step_size = step_size_ms * units.ms 

    idles = [sycomore.bloch.time_interval(species, step_size) for species in species_ensemble]

    pulse_excitation = sycomore.bloch.pulse(flip_angle_deg * units.deg, phase=np.pi*units.rad)
    pulse_refocus = sycomore.bloch.pulse(180 * units.deg, phase=np.pi*units.rad/2) 
    pulse_tipup = sycomore.bloch.pulse(-90 * units.deg, phase=np.pi*units.rad) 

    t = 0 * units.ms
    M = [np.array([0, 0, 1, 1]) for _ in range(n_species)]
    records = [[[t.convert_to(units.ms), m[:3] / m[3]]] for m in M]

    for k, m in enumerate(M):
        M[k] = update_and_record_magnetization(m, records[k], t, update=pulse_excitation)

    for step in range(n_steps):
        if (step != step_refocus)&(step != 2*step_refocus): 
            t += step_size
            for k, m in enumerate(M):
                M[k] = update_and_record_magnetization(m, records[k], t, update=idles[k])
        elif step == step_refocus:
            for k, m in enumerate(M):
                M[k] = update_and_record_magnetization(m, records[k], t, update=pulse_refocus)
        elif step == (step_refocus * 2):
            for k, m in enumerate(M):
                M[k] = update_and_record_magnetization(m, records[k], t, update=pulse_tipup)
    
    magnetization = np.stack([np.array([m for _, m in record]) for record in records]).mean(axis=0)
    magnetization_k = np.array([m for _, m in records[0]])
    time = np.array([t for t, _ in records[0]])

    return magnetization, time, magnetization_k
