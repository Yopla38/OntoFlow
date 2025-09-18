"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# test/test_phase_3_complete_multifiles.py
"""
Test complet Phase 3 avec projet multi-fichiers Fortran réaliste.
Simule un vrai projet de dynamique moléculaire avec dépendances complexes.
"""

import asyncio
import re
import tempfile
import os
from pathlib import Path
from typing import List

# === FICHIERS FORTRAN DE TEST ===

FORTRAN_FILES = {
    'constants.f90': """
module constants
    use iso_fortran_env, only: real64
    implicit none

    ! Physical constants
    real(real64), parameter :: kb = 1.380649e-23_real64     ! Boltzmann constant [J/K]
    real(real64), parameter :: na = 6.02214076e23_real64    ! Avogadro number [mol^-1]
    real(real64), parameter :: pi = 3.141592653589793_real64
    real(real64), parameter :: eps0 = 8.8541878128e-12_real64 ! Vacuum permittivity

    ! Simulation parameters
    integer, parameter :: max_particles = 100000
    real(real64), parameter :: default_dt = 0.001_real64    ! Default timestep [ps]
    real(real64), parameter :: cutoff_distance = 2.5_real64 ! LJ cutoff [sigma]

    ! Unit conversions
    real(real64), parameter :: amu_to_kg = 1.66054e-27_real64
    real(real64), parameter :: ang_to_m = 1.0e-10_real64

end module constants
    """,

    'types.f90': """
module types
    use iso_fortran_env, only: real64
    implicit none

    ! Particle type definition
    type :: particle_t
        real(real64) :: x, y, z           ! Position [Angstrom]
        real(real64) :: vx, vy, vz        ! Velocity [Angstrom/ps]  
        real(real64) :: fx, fy, fz        ! Force [amu*Angstrom/ps^2]
        real(real64) :: mass              ! Mass [amu]
        integer :: id                     ! Particle ID
        integer :: type_id                ! Particle type (Ar=1, Ne=2, etc.)
    end type particle_t

    ! System state type
    type :: system_t
        type(particle_t), allocatable :: particles(:)
        integer :: n_particles
        real(real64) :: box_size(3)       ! Box dimensions [Angstrom]
        real(real64) :: temperature       ! Temperature [K]
        real(real64) :: pressure          ! Pressure [atm]
        real(real64) :: total_energy      ! Total energy [K]
        real(real64) :: kinetic_energy    ! Kinetic energy [K]
        real(real64) :: potential_energy  ! Potential energy [K]
        logical :: use_pbc                ! Periodic boundary conditions
    end type system_t

    ! Simulation parameters type
    type :: simulation_params_t
        real(real64) :: dt                ! Timestep [ps]
        integer :: nsteps                 ! Number of steps
        integer :: nprint                 ! Print frequency
        integer :: nsave                  ! Save frequency
        character(len=256) :: output_file ! Output filename
        logical :: thermostat_on          ! Use thermostat
        real(real64) :: target_temp       ! Target temperature [K]
        real(real64) :: tau_temp          ! Temperature coupling time [ps]
    end type simulation_params_t

end module types
    """,

    'math_utils.f90': """
module math_utils
    use iso_fortran_env, only: real64
    use constants, only: pi
    implicit none

    private
    public :: distance, distance_squared, normalize_vector
    public :: random_gaussian, random_uniform, cross_product
    public :: apply_minimum_image, wrap_coordinate

    contains

    pure function distance(x1, y1, z1, x2, y2, z2) result(dist)
        real(real64), intent(in) :: x1, y1, z1, x2, y2, z2
        real(real64) :: dist

        dist = sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
    end function distance

    pure function distance_squared(x1, y1, z1, x2, y2, z2) result(dist2)
        real(real64), intent(in) :: x1, y1, z1, x2, y2, z2
        real(real64) :: dist2

        dist2 = (x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2
    end function distance_squared

    subroutine normalize_vector(vx, vy, vz)
        real(real64), intent(inout) :: vx, vy, vz
        real(real64) :: norm

        norm = sqrt(vx**2 + vy**2 + vz**2)
        if (norm > 1e-12_real64) then
            vx = vx / norm
            vy = vy / norm  
            vz = vz / norm
        end if
    end subroutine normalize_vector

    function random_gaussian(mean, std_dev) result(value)
        real(real64), intent(in) :: mean, std_dev
        real(real64) :: value
        real(real64) :: u1, u2

        call random_number(u1)
        call random_number(u2)

        ! Box-Muller transformation
        value = mean + std_dev * sqrt(-2.0_real64 * log(u1)) * cos(2.0_real64 * pi * u2)
    end function random_gaussian

    function random_uniform(min_val, max_val) result(value)
        real(real64), intent(in) :: min_val, max_val
        real(real64) :: value, r

        call random_number(r)
        value = min_val + r * (max_val - min_val)
    end function random_uniform

    pure function cross_product(a, b) result(c)
        real(real64), intent(in) :: a(3), b(3)
        real(real64) :: c(3)

        c(1) = a(2) * b(3) - a(3) * b(2)
        c(2) = a(3) * b(1) - a(1) * b(3)
        c(3) = a(1) * b(2) - a(2) * b(1)
    end function cross_product

    pure function apply_minimum_image(dr, box_size) result(dr_pbc)
        real(real64), intent(in) :: dr, box_size
        real(real64) :: dr_pbc

        dr_pbc = dr - box_size * nint(dr / box_size)
    end function apply_minimum_image

    pure function wrap_coordinate(coord, box_size) result(wrapped)
        real(real64), intent(in) :: coord, box_size
        real(real64) :: wrapped

        wrapped = coord - box_size * floor(coord / box_size)
    end function wrap_coordinate

end module math_utils
    """,

    'forces.f90': """
module forces
    use iso_fortran_env, only: real64
    use types, only: particle_t, system_t
    use constants, only: cutoff_distance
    use math_utils, only: distance_squared, apply_minimum_image
    implicit none

    private
    public :: compute_forces, lennard_jones_force, compute_potential_energy
    public :: zero_forces, apply_periodic_boundary_forces

    contains

    subroutine compute_forces(system)
        type(system_t), intent(inout) :: system
        integer :: i, j
        real(real64) :: dx, dy, dz, r2, force_mag
        real(real64) :: fx_ij, fy_ij, fz_ij

        ! Zero all forces
        call zero_forces(system)

        ! Pair interactions
        do i = 1, system%n_particles - 1
            do j = i + 1, system%n_particles

                ! Calculate distance with PBC
                dx = system%particles(j)%x - system%particles(i)%x
                dy = system%particles(j)%y - system%particles(i)%y 
                dz = system%particles(j)%z - system%particles(i)%z

                if (system%use_pbc) then
                    dx = apply_minimum_image(dx, system%box_size(1))
                    dy = apply_minimum_image(dy, system%box_size(2))
                    dz = apply_minimum_image(dz, system%box_size(3))
                end if

                r2 = dx*dx + dy*dy + dz*dz

                if (r2 < cutoff_distance**2) then
                    call lennard_jones_force(r2, force_mag)

                    fx_ij = force_mag * dx
                    fy_ij = force_mag * dy
                    fz_ij = force_mag * dz

                    ! Newton's third law
                    system%particles(i)%fx = system%particles(i)%fx - fx_ij
                    system%particles(i)%fy = system%particles(i)%fy - fy_ij
                    system%particles(i)%fz = system%particles(i)%fz - fz_ij

                    system%particles(j)%fx = system%particles(j)%fx + fx_ij
                    system%particles(j)%fy = system%particles(j)%fy + fy_ij
                    system%particles(j)%fz = system%particles(j)%fz + fz_ij
                end if

            end do
        end do

    end subroutine compute_forces

    pure subroutine lennard_jones_force(r2, force_magnitude)
        real(real64), intent(in) :: r2
        real(real64), intent(out) :: force_magnitude
        real(real64) :: r6, r12, epsilon, sigma2

        ! LJ parameters for Argon
        epsilon = 119.8_real64  ! [K]
        sigma2 = 3.405_real64**2 ! [Angstrom^2]

        r6 = (sigma2 / r2)**3
        r12 = r6 * r6

        force_magnitude = 24.0_real64 * epsilon * (2.0_real64 * r12 - r6) / r2
    end subroutine lennard_jones_force

    subroutine zero_forces(system)
        type(system_t), intent(inout) :: system
        integer :: i

        do i = 1, system%n_particles
            system%particles(i)%fx = 0.0_real64
            system%particles(i)%fy = 0.0_real64
            system%particles(i)%fz = 0.0_real64
        end do
    end subroutine zero_forces

    function compute_potential_energy(system) result(potential)
        type(system_t), intent(in) :: system
        real(real64) :: potential
        integer :: i, j
        real(real64) :: dx, dy, dz, r2, r6, r12
        real(real64) :: epsilon, sigma2

        epsilon = 119.8_real64
        sigma2 = 3.405_real64**2
        potential = 0.0_real64

        do i = 1, system%n_particles - 1
            do j = i + 1, system%n_particles
                dx = system%particles(j)%x - system%particles(i)%x
                dy = system%particles(j)%y - system%particles(i)%y
                dz = system%particles(j)%z - system%particles(i)%z

                if (system%use_pbc) then
                    dx = apply_minimum_image(dx, system%box_size(1))
                    dy = apply_minimum_image(dy, system%box_size(2))
                    dz = apply_minimum_image(dz, system%box_size(3))
                end if

                r2 = dx*dx + dy*dy + dz*dz

                if (r2 < cutoff_distance**2) then
                    r6 = (sigma2 / r2)**3
                    r12 = r6 * r6
                    potential = potential + 4.0_real64 * epsilon * (r12 - r6)
                end if
            end do
        end do

    end function compute_potential_energy

    subroutine apply_periodic_boundary_forces(system)
        type(system_t), intent(inout) :: system
        integer :: i

        ! Apply PBC to particle positions
        do i = 1, system%n_particles
            system%particles(i)%x = wrap_coordinate(system%particles(i)%x, system%box_size(1))
            system%particles(i)%y = wrap_coordinate(system%particles(i)%y, system%box_size(2))
            system%particles(i)%z = wrap_coordinate(system%particles(i)%z, system%box_size(3))
        end do

        contains
            pure function wrap_coordinate(coord, box_size) result(wrapped)
                real(real64), intent(in) :: coord, box_size
                real(real64) :: wrapped
                wrapped = coord - box_size * floor(coord / box_size)
            end function wrap_coordinate

    end subroutine apply_periodic_boundary_forces

end module forces
    """,

    'integrator.f90': """
module integrator
    use iso_fortran_env, only: real64
    use types, only: particle_t, system_t
    use constants, only: kb
    use forces, only: compute_forces, compute_potential_energy
    use math_utils, only: random_gaussian
    implicit none

    private
    public :: velocity_verlet_step, compute_kinetic_energy, compute_temperature
    public :: initialize_velocities, thermostat_berendsen, remove_com_motion

    contains

    subroutine velocity_verlet_step(system, dt)
        type(system_t), intent(inout) :: system
        real(real64), intent(in) :: dt
        integer :: i
        real(real64) :: dt2_half

        dt2_half = 0.5_real64 * dt * dt

        ! First half of velocity update and position update
        do i = 1, system%n_particles
            ! v(t + dt/2) = v(t) + (dt/2) * F(t) / m
            system%particles(i)%vx = system%particles(i)%vx + &
                                    0.5_real64 * dt * system%particles(i)%fx / system%particles(i)%mass
            system%particles(i)%vy = system%particles(i)%vy + &
                                    0.5_real64 * dt * system%particles(i)%fy / system%particles(i)%mass
            system%particles(i)%vz = system%particles(i)%vz + &
                                    0.5_real64 * dt * system%particles(i)%fz / system%particles(i)%mass

            ! r(t + dt) = r(t) + dt * v(t + dt/2)
            system%particles(i)%x = system%particles(i)%x + dt * system%particles(i)%vx
            system%particles(i)%y = system%particles(i)%y + dt * system%particles(i)%vy
            system%particles(i)%z = system%particles(i)%z + dt * system%particles(i)%vz
        end do

        ! Compute new forces F(t + dt)
        call compute_forces(system)

        ! Second half of velocity update
        do i = 1, system%n_particles
            ! v(t + dt) = v(t + dt/2) + (dt/2) * F(t + dt) / m
            system%particles(i)%vx = system%particles(i)%vx + &
                                    0.5_real64 * dt * system%particles(i)%fx / system%particles(i)%mass
            system%particles(i)%vy = system%particles(i)%vy + &
                                    0.5_real64 * dt * system%particles(i)%fy / system%particles(i)%mass
            system%particles(i)%vz = system%particles(i)%vz + &
                                    0.5_real64 * dt * system%particles(i)%fz / system%particles(i)%mass
        end do

        ! Update energies
        system%kinetic_energy = compute_kinetic_energy(system)
        system%potential_energy = compute_potential_energy(system)
        system%total_energy = system%kinetic_energy + system%potential_energy
        system%temperature = compute_temperature(system)

    end subroutine velocity_verlet_step

    pure function compute_kinetic_energy(system) result(kinetic)
        type(system_t), intent(in) :: system
        real(real64) :: kinetic
        integer :: i

        kinetic = 0.0_real64
        do i = 1, system%n_particles
            kinetic = kinetic + 0.5_real64 * system%particles(i)%mass * &
                     (system%particles(i)%vx**2 + system%particles(i)%vy**2 + system%particles(i)%vz**2)
        end do
    end function compute_kinetic_energy

    pure function compute_temperature(system) result(temperature)
        type(system_t), intent(in) :: system
        real(real64) :: temperature

        ! T = (2/3) * KE / (N * kb) for 3D system
        temperature = (2.0_real64 / 3.0_real64) * system%kinetic_energy / &
                     (real(system%n_particles, real64) * kb)
    end function compute_temperature

    subroutine initialize_velocities(system, target_temperature)
        type(system_t), intent(inout) :: system
        real(real64), intent(in) :: target_temperature
        integer :: i
        real(real64) :: sigma_v

        ! Maxwell-Boltzmann distribution
        do i = 1, system%n_particles
            sigma_v = sqrt(kb * target_temperature / system%particles(i)%mass)

            system%particles(i)%vx = random_gaussian(0.0_real64, sigma_v)
            system%particles(i)%vy = random_gaussian(0.0_real64, sigma_v)
            system%particles(i)%vz = random_gaussian(0.0_real64, sigma_v)
        end do

        ! Remove center of mass motion
        call remove_com_motion(system)

        ! Scale to exact target temperature
        call scale_velocities_to_temperature(system, target_temperature)

    end subroutine initialize_velocities

    subroutine thermostat_berendsen(system, target_temp, tau, dt)
        type(system_t), intent(inout) :: system
        real(real64), intent(in) :: target_temp, tau, dt
        real(real64) :: current_temp, scaling_factor
        integer :: i

        current_temp = compute_temperature(system)

        if (current_temp > 0.0_real64) then
            scaling_factor = sqrt(1.0_real64 + (dt / tau) * (target_temp / current_temp - 1.0_real64))

            do i = 1, system%n_particles
                system%particles(i)%vx = system%particles(i)%vx * scaling_factor
                system%particles(i)%vy = system%particles(i)%vy * scaling_factor
                system%particles(i)%vz = system%particles(i)%vz * scaling_factor
            end do

            system%kinetic_energy = system%kinetic_energy * scaling_factor**2
            system%temperature = compute_temperature(system)
        end if

    end subroutine thermostat_berendsen

    subroutine remove_com_motion(system)
        type(system_t), intent(inout) :: system
        real(real64) :: total_mass, com_vx, com_vy, com_vz
        integer :: i

        ! Calculate center of mass velocity
        total_mass = 0.0_real64
        com_vx = 0.0_real64
        com_vy = 0.0_real64
        com_vz = 0.0_real64

        do i = 1, system%n_particles
            total_mass = total_mass + system%particles(i)%mass
            com_vx = com_vx + system%particles(i)%mass * system%particles(i)%vx
            com_vy = com_vy + system%particles(i)%mass * system%particles(i)%vy
            com_vz = com_vz + system%particles(i)%mass * system%particles(i)%vz
        end do

        com_vx = com_vx / total_mass
        com_vy = com_vy / total_mass
        com_vz = com_vz / total_mass

        ! Remove COM motion
        do i = 1, system%n_particles
            system%particles(i)%vx = system%particles(i)%vx - com_vx
            system%particles(i)%vy = system%particles(i)%vy - com_vy
            system%particles(i)%vz = system%particles(i)%vz - com_vz
        end do

    end subroutine remove_com_motion

    subroutine scale_velocities_to_temperature(system, target_temp)
        type(system_t), intent(inout) :: system
        real(real64), intent(in) :: target_temp
        real(real64) :: current_temp, scaling_factor
        integer :: i

        current_temp = compute_temperature(system)

        if (current_temp > 0.0_real64) then
            scaling_factor = sqrt(target_temp / current_temp)

            do i = 1, system%n_particles
                system%particles(i)%vx = system%particles(i)%vx * scaling_factor
                system%particles(i)%vy = system%particles(i)%vy * scaling_factor
                system%particles(i)%vz = system%particles(i)%vz * scaling_factor
            end do
        end if

    end subroutine scale_velocities_to_temperature

end module integrator
    """,

    'simulation.f90': """
module simulation
    use iso_fortran_env, only: real64
    use types, only: particle_t, system_t, simulation_params_t
    use constants, only: default_dt, amu_to_kg, ang_to_m
    use forces, only: compute_forces
    use integrator, only: velocity_verlet_step, initialize_velocities, thermostat_berendsen
    use math_utils, only: random_uniform
    implicit none

    private
    public :: initialize_system, run_md_simulation, setup_fcc_lattice
    public :: compute_rdf, save_trajectory, print_system_info

    contains

    subroutine initialize_system(system, n_particles, density, temperature)
        type(system_t), intent(out) :: system
        integer, intent(in) :: n_particles
        real(real64), intent(in) :: density, temperature
        real(real64) :: volume, box_length

        system%n_particles = n_particles
        system%temperature = temperature
        system%use_pbc = .true.

        ! Calculate box size for given density
        volume = real(n_particles, real64) / density
        box_length = volume**(1.0_real64/3.0_real64)
        system%box_size = [box_length, box_length, box_length]

        ! Allocate particle array
        allocate(system%particles(n_particles))

        ! Initialize particle properties
        call setup_argon_particles(system)

        ! Setup initial configuration
        call setup_fcc_lattice(system)

        ! Initialize velocities
        call initialize_velocities(system, temperature)

        ! Compute initial forces and energies
        call compute_forces(system)

        print *, "System initialized:"
        call print_system_info(system)

    end subroutine initialize_system

    subroutine setup_argon_particles(system)
        type(system_t), intent(inout) :: system
        integer :: i
        real(real64) :: argon_mass

        argon_mass = 39.948_real64  ! amu

        do i = 1, system%n_particles
            system%particles(i)%id = i
            system%particles(i)%type_id = 1  ! Argon
            system%particles(i)%mass = argon_mass
        end do

    end subroutine setup_argon_particles

    subroutine setup_fcc_lattice(system)
        type(system_t), intent(inout) :: system
        integer :: nx, ny, nz, i, ix, iy, iz, particle_count
        real(real64) :: lattice_constant, x, y, z
        real(real64) :: fcc_positions(4,3)

        ! Estimate lattice size
        nx = ceiling((real(system%n_particles, real64) / 4.0_real64)**(1.0_real64/3.0_real64))
        ny = nx
        nz = nx

        lattice_constant = system%box_size(1) / real(nx, real64)

        ! FCC basis vectors
        fcc_positions(1,:) = [0.0_real64, 0.0_real64, 0.0_real64]
        fcc_positions(2,:) = [0.5_real64, 0.5_real64, 0.0_real64]
        fcc_positions(3,:) = [0.5_real64, 0.0_real64, 0.5_real64]
        fcc_positions(4,:) = [0.0_real64, 0.5_real64, 0.5_real64]

        particle_count = 0

        do ix = 0, nx-1
            do iy = 0, ny-1
                do iz = 0, nz-1
                    do i = 1, 4
                        if (particle_count >= system%n_particles) exit

                        particle_count = particle_count + 1

                        x = (real(ix, real64) + fcc_positions(i,1)) * lattice_constant
                        y = (real(iy, real64) + fcc_positions(i,2)) * lattice_constant
                        z = (real(iz, real64) + fcc_positions(i,3)) * lattice_constant

                        system%particles(particle_count)%x = x
                        system%particles(particle_count)%y = y
                        system%particles(particle_count)%z = z
                    end do
                    if (particle_count >= system%n_particles) exit
                end do
                if (particle_count >= system%n_particles) exit
            end do
            if (particle_count >= system%n_particles) exit
        end do

    end subroutine setup_fcc_lattice

    subroutine run_md_simulation(system, params)
        type(system_t), intent(inout) :: system
        type(simulation_params_t), intent(in) :: params
        integer :: step
        real(real64) :: start_time, end_time

        call cpu_time(start_time)

        print *, "Starting MD simulation..."
        print *, "Steps:", params%nsteps
        print *, "Timestep:", params%dt, "ps"
        print *, "Target temperature:", params%target_temp, "K"

        do step = 1, params%nsteps

            ! Integration step
            call velocity_verlet_step(system, params%dt)

            ! Apply thermostat if enabled
            if (params%thermostat_on) then
                call thermostat_berendsen(system, params%target_temp, params%tau_temp, params%dt)
            end if

            ! Print progress
            if (mod(step, params%nprint) == 0) then
                print *, "Step:", step, "T =", system%temperature, "K", &
                        "E_total =", system%total_energy, "K"
            end if

            ! Save trajectory
            if (mod(step, params%nsave) == 0) then
                call save_trajectory(system, step, params%output_file)
            end if

        end do

        call cpu_time(end_time)

        print *, "Simulation completed in", end_time - start_time, "seconds"
        call print_final_statistics(system, params)

    end subroutine run_md_simulation

    subroutine print_system_info(system)
        type(system_t), intent(in) :: system

        print *, "N particles:", system%n_particles
        print *, "Box size:", system%box_size
        print *, "Temperature:", system%temperature, "K"
        print *, "Kinetic energy:", system%kinetic_energy, "K"
        print *, "Potential energy:", system%potential_energy, "K"
        print *, "Total energy:", system%total_energy, "K"
        print *, "PBC enabled:", system%use_pbc

    end subroutine print_system_info

    subroutine print_final_statistics(system, params)
        type(system_t), intent(in) :: system
        type(simulation_params_t), intent(in) :: params
        real(real64) :: performance

        print *, ""
        print *, "=== FINAL STATISTICS ==="
        call print_system_info(system)

        performance = real(params%nsteps * system%n_particles, real64) / 1000.0_real64
        print *, "Performance: ~", performance, "k particle-steps"

    end subroutine print_final_statistics

    subroutine save_trajectory(system, step, filename)
        type(system_t), intent(in) :: system
        integer, intent(in) :: step
        character(len=*), intent(in) :: filename
        integer :: unit, i

        ! Simple XYZ format output
        open(newunit=unit, file=trim(filename), position='append')

        write(unit, '(I0)') system%n_particles
        write(unit, '(A,I0,A,F10.3,A)') 'Step ', step, ' T=', system%temperature, 'K'

        do i = 1, system%n_particles
            write(unit, '(A,3F12.6)') 'Ar', system%particles(i)%x, &
                                              system%particles(i)%y, &
                                              system%particles(i)%z
        end do

        close(unit)

    end subroutine save_trajectory

    function compute_rdf(system, max_r, nbins) result(rdf)
        type(system_t), intent(in) :: system
        real(real64), intent(in) :: max_r
        integer, intent(in) :: nbins
        real(real64) :: rdf(nbins)

        ! Simplified RDF calculation
        integer :: i, j, bin_idx
        real(real64) :: dr, r, density, volume
        integer :: histogram(nbins)

        dr = max_r / real(nbins, real64)
        histogram = 0

        do i = 1, system%n_particles - 1
            do j = i + 1, system%n_particles
                r = sqrt((system%particles(j)%x - system%particles(i)%x)**2 + &
                        (system%particles(j)%y - system%particles(i)%y)**2 + &
                        (system%particles(j)%z - system%particles(i)%z)**2)

                if (r < max_r) then
                    bin_idx = int(r / dr) + 1
                    if (bin_idx <= nbins) then
                        histogram(bin_idx) = histogram(bin_idx) + 1
                    end if
                end if
            end do
        end do

        ! Normalize
        volume = product(system%box_size)
        density = real(system%n_particles, real64) / volume

        do i = 1, nbins
            r = (real(i, real64) - 0.5_real64) * dr
            rdf(i) = real(histogram(i), real64) * volume / &
                    (4.0_real64 * 3.14159_real64 * r**2 * dr * real(system%n_particles, real64) * density)
        end do

    end function compute_rdf

end module simulation
    """,

    'main.f90': """
program molecular_dynamics_main
    use iso_fortran_env, only: real64
    use types, only: system_t, simulation_params_t
    use simulation, only: initialize_system, run_md_simulation
    implicit none

    type(system_t) :: md_system
    type(simulation_params_t) :: sim_params

    ! System parameters
    integer, parameter :: n_particles = 864  ! 6x6x6 FCC unit cells
    real(real64), parameter :: density = 0.8_real64  ! Reduced density
    real(real64), parameter :: temperature = 94.4_real64  ! Kelvin (T* = 0.8)

    ! Simulation parameters
    sim_params%dt = 0.002_real64          ! 2 fs timestep
    sim_params%nsteps = 10000              ! 20 ps simulation
    sim_params%nprint = 500                ! Print every 1 ps
    sim_params%nsave = 1000                ! Save every 2 ps
    sim_params%output_file = 'trajectory.xyz'
    sim_params%thermostat_on = .true.
    sim_params%target_temp = temperature
    sim_params%tau_temp = 0.1_real64       ! 100 fs coupling time

    print *, "============================================"
    print *, "    MOLECULAR DYNAMICS SIMULATION"
    print *, "      Lennard-Jones Argon System"
    print *, "============================================"
    print *, ""

    ! Initialize system
    call initialize_system(md_system, n_particles, density, temperature)

    ! Run simulation
    call run_md_simulation(md_system, sim_params)

    print *, ""
    print *, "Simulation completed successfully!"
    print *, "Trajectory saved to:", trim(sim_params%output_file)

end program molecular_dynamics_main
    """
}


async def test_complete_multifile_system():
    """Test complet avec système multi-fichiers réaliste"""

    print("🏗️ TEST COMPLET MULTI-FICHIERS PHASE 3")
    print("=" * 60)
    print("Simulation d'un projet de dynamique moléculaire complet")
    print("=" * 60)

    class CompleteMockStore:
        def __init__(self, fortran_files: dict):
            self.docs = {}
            self.chunks = {}

            # Créer les documents et chunks depuis les fichiers Fortran
            for filename, code in fortran_files.items():
                doc_id = filename.replace('.f90', '')
                self.docs[doc_id] = {'id': doc_id, 'path': filename}

                # Analyser le code pour créer des chunks réalistes
                self.chunks[doc_id] = self._analyze_fortran_file(filename, code, doc_id)

        def _analyze_fortran_file(self, filename: str, code: str, doc_id: str) -> List[dict]:
            """Analyse un fichier Fortran pour créer des chunks réalistes"""
            chunks = []
            lines = code.split('\n')

            # Patterns pour détecter les entités
            entity_patterns = {
                'module': re.compile(r'^\s*module\s+(\w+)', re.IGNORECASE),
                'program': re.compile(r'^\s*program\s+(\w+)', re.IGNORECASE),
                'subroutine': re.compile(r'^\s*subroutine\s+(\w+)', re.IGNORECASE),
                'function': re.compile(r'^\s*(?:pure\s+|elemental\s+)*(?:real|integer|logical)*.*?function\s+(\w+)',
                                       re.IGNORECASE),
                'type': re.compile(r'^\s*type\s*(?:::)?\s*(\w+)', re.IGNORECASE)
            }

            # Extraire les dépendances USE
            dependencies = []
            use_pattern = re.compile(r'^\s*use\s+(\w+)', re.IGNORECASE)
            for line in lines:
                match = use_pattern.match(line)
                if match:
                    dependencies.append(match.group(1))

            current_entity = None
            entity_start = 0
            chunk_index = 0

            for i, line in enumerate(lines):
                line_stripped = line.strip()

                # Détecter début d'entité
                for entity_type, pattern in entity_patterns.items():
                    match = pattern.match(line)
                    if match:
                        # Finaliser l'entité précédente si elle existe
                        if current_entity:
                            self._finalize_entity_chunk(
                                chunks, current_entity, lines, entity_start, i - 1,
                                dependencies, doc_id, chunk_index, filename
                            )
                            chunk_index += 1

                        # Commencer nouvelle entité
                        current_entity = {
                            'name': match.group(1),
                            'type': entity_type,
                            'start_line': i + 1
                        }
                        entity_start = i
                        break

                # Détecter fin d'entité
                if line_stripped.startswith('end ') and current_entity:
                    self._finalize_entity_chunk(
                        chunks, current_entity, lines, entity_start, i,
                        dependencies, doc_id, chunk_index, filename
                    )
                    current_entity = None
                    chunk_index += 1

            # Finaliser la dernière entité
            if current_entity:
                self._finalize_entity_chunk(
                    chunks, current_entity, lines, entity_start, len(lines) - 1,
                    dependencies, doc_id, chunk_index, filename
                )

            return chunks

        def _finalize_entity_chunk(self, chunks, entity, lines, start_idx, end_idx,
                                   dependencies, doc_id, chunk_index, filename):
            """Finalise un chunk d'entité"""
            entity_code = '\n'.join(lines[start_idx:end_idx + 1])

            # Détecter les concepts selon le contenu
            concepts = self._detect_concepts(entity['name'], entity['type'], entity_code)

            chunk = {
                'id': f"{doc_id}-chunk-{chunk_index}",
                'text': entity_code,
                'metadata': {
                    'entity_name': entity['name'],
                    'entity_type': entity['type'],
                    'filepath': filename,
                    'filename': filename,
                    'start_pos': entity['start_line'],
                    'end_pos': end_idx + 1,
                    'dependencies': dependencies.copy(),
                    'detected_concepts': concepts
                }
            }
            chunks.append(chunk)

        def _detect_concepts(self, name: str, entity_type: str, code: str) -> List[dict]:
            """Détecte les concepts dans le code"""
            concepts = []
            name_lower = name.lower()
            code_lower = code.lower()

            # Concepts par nom et contenu
            concept_patterns = {
                'molecular_dynamics': ['molecular', 'dynamics', 'md', 'simulation'],
                'force_calculation': ['force', 'lennard', 'jones', 'potential'],
                'numerical_integration': ['verlet', 'integrat', 'timestep', 'velocity'],
                'thermodynamics': ['temperature', 'thermostat', 'kinetic', 'energy'],
                'mathematics': ['sqrt', 'sin', 'cos', 'random', 'gaussian'],
                'linear_algebra': ['vector', 'cross_product', 'normalize'],
                'periodic_boundary': ['periodic', 'boundary', 'pbc', 'minimum_image'],
                'initialization': ['init', 'setup', 'create'],
                'io_operations': ['write', 'read', 'save', 'trajectory'],
                'data_structures': ['type', 'particle', 'system']
            }

            for concept, keywords in concept_patterns.items():
                confidence = 0.0
                matched_keywords = []

                for keyword in keywords:
                    if keyword in name_lower:
                        confidence += 0.3
                        matched_keywords.append(keyword)
                    if keyword in code_lower:
                        confidence += 0.1
                        matched_keywords.append(keyword)

                if confidence > 0.2:
                    concepts.append({
                        'label': concept,
                        'confidence': min(confidence, 1.0),
                        'category': 'physics' if concept in ['molecular_dynamics', 'force_calculation',
                                                             'thermodynamics'] else 'algorithm',
                        'keywords': matched_keywords[:3]
                    })

            return concepts[:5]  # Top 5 concepts

        async def get_all_documents(self):
            return list(self.docs.keys())

        async def get_document_chunks(self, doc_id):
            return self.chunks.get(doc_id, [])

        async def load_document_chunks(self, doc_id):
            return True

    class MockRAGEngine:
        async def find_similar(self, text, max_results=10, min_similarity=0.6):
            # Simuler des résultats de similarité basés sur le contenu
            results = []
            if 'force' in text.lower():
                results.append(('forces-chunk-1', 0.85))
                results.append(('integrator-chunk-2', 0.72))
            elif 'velocity' in text.lower():
                results.append(('integrator-chunk-0', 0.88))
                results.append(('simulation-chunk-1', 0.75))
            elif 'temperature' in text.lower():
                results.append(('integrator-chunk-1', 0.82))
                results.append(('simulation-chunk-0', 0.68))
            return results[:max_results]

    # === DÉBUT DES TESTS ===

    try:
        print("\n1. 🏗️ Setup du système multi-fichiers...")

        mock_store = CompleteMockStore(FORTRAN_FILES)
        mock_rag = MockRAGEngine()

        print(f"   ✅ {len(FORTRAN_FILES)} fichiers Fortran chargés")
        print(f"   📄 Fichiers: {list(FORTRAN_FILES.keys())}")

        total_chunks = sum(len(chunks) for chunks in mock_store.chunks.values())
        print(f"   🧩 {total_chunks} chunks générés au total")

        # Stats par fichier
        for filename, chunks in mock_store.chunks.items():
            print(f"      {filename}: {len(chunks)} entités")

        print("\n2. 🚀 Initialisation de l'orchestrateur...")

        from ..providers.smart_orchestrator import SmartContextOrchestrator

        orchestrator = SmartContextOrchestrator(mock_store, mock_rag)
        await orchestrator.initialize()

        stats = orchestrator.get_index_stats()
        print(f"   ✅ EntityManager: {stats['total_entities']} entités indexées")
        print(f"   📊 Types: {stats['entity_types']}")
        print(f"   📁 Fichiers: {stats['files_indexed']}")
        print(f"   📦 Entités regroupées: {stats['grouped_entities']}")

        print("\n3. 🎯 Tests par type d'agent sur entités clés...")

        # Entités importantes à tester
        key_entities = [
            'forces',  # Module central
            'velocity_verlet_step',  # Algorithme critique
            'lennard_jones_force',  # Calcul physique
            'molecular_dynamics_main',  # Programme principal
            'particle_t'  # Type de données
        ]

        agent_scenarios = [
            ("developer", "debugging"),
            ("reviewer", "performance_review"),
            ("analyzer", "dependency_analysis"),
            ("documenter", "api_documentation")
        ]

        test_results = {}

        for entity_name in key_entities:
            print(f"\n   🔍 Test entité: {entity_name}")
            test_results[entity_name] = {}

            for agent_type, task_context in agent_scenarios:
                try:
                    context = await orchestrator.get_context_for_agent(
                        entity_name, agent_type, task_context, 3000
                    )

                    # Analyser les résultats
                    tokens = context['total_tokens']
                    contexts_generated = list(context['contexts'].keys())
                    errors = [k for k, v in context['contexts'].items()
                              if isinstance(v, dict) and 'error' in v]
                    insights_count = len(context.get('key_insights', []))
                    recs_count = len(context.get('recommendations', []))
                    agent_insights = len(context.get('agent_specific_insights', []))

                    test_results[entity_name][f"{agent_type}_{task_context}"] = {
                        'success': len(errors) == 0,
                        'tokens': tokens,
                        'contexts': contexts_generated,
                        'errors': errors,
                        'insights': insights_count,
                        'recommendations': recs_count,
                        'agent_insights': agent_insights
                    }

                    status = "✅" if len(errors) == 0 else f"⚠️ ({len(errors)} erreurs)"
                    print(f"      {agent_type}/{task_context}: {status} {tokens}t, {insights_count}i, {recs_count}r")

                except Exception as e:
                    test_results[entity_name][f"{agent_type}_{task_context}"] = {
                        'success': False,
                        'error': str(e)
                    }
                    print(f"      {agent_type}/{task_context}: ❌ Exception: {str(e)[:50]}...")

                    print("\n4. 📊 Analyse des dépendances inter-modules...")

                    # Tester les dépendances complexes
                    dependency_tests = [
                        ('simulation', 'forces'),
                        ('integrator', 'math_utils'),
                        ('forces', 'constants'),
                        ('molecular_dynamics_main', 'simulation')
                    ]

                    for source, target in dependency_tests:
                        try:
                            # Test contexte global pour analyser les dépendances
                            dep_context = await orchestrator.get_global_context(source, 2000)

                            if 'error' not in dep_context:
                                # Chercher si la dépendance est détectée
                                deps_found = []
                                if 'impact_analysis' in dep_context:
                                    deps_found.extend(dep_context['impact_analysis'].get('direct_dependents', []))

                                modules_info = dep_context.get('module_hierarchy', {}).get('modules', {})
                                if source in modules_info:
                                    deps_found.extend(modules_info[source].get('dependencies', []))

                                dependency_detected = target in deps_found or any(
                                    target.lower() in dep.lower() for dep in deps_found)
                                status = "✅" if dependency_detected else "⚠️"
                                print(
                                    f"   {status} {source} → {target}: {'détectée' if dependency_detected else 'non détectée'}")
                            else:
                                print(f"   ❌ {source} → {target}: erreur analyse")

                        except Exception as e:
                            print(f"   ❌ {source} → {target}: exception {str(e)[:30]}...")

                    print("\n5. 🧠 Test détection de concepts et patterns...")

                    # Entités avec patterns algorithmiques attendus
                    pattern_tests = [
                        ('velocity_verlet_step', ['numerical_integration', 'molecular_dynamics']),
                        ('lennard_jones_force', ['force_calculation', 'physics']),
                        ('random_gaussian', ['mathematics', 'statistics']),
                        ('compute_rdf', ['data_analysis', 'physics'])
                    ]

                    concept_success = 0
                    total_concept_tests = 0

                    for entity_name, expected_concepts in pattern_tests:
                        try:
                            semantic_context = await orchestrator.get_semantic_context(entity_name, 2000)

                            if 'error' not in semantic_context:
                                detected_concepts = semantic_context.get('main_concepts', [])
                                concept_labels = [c.get('label', '') for c in detected_concepts]

                                algorithmic_patterns = semantic_context.get('algorithmic_patterns', [])
                                pattern_labels = [p.get('pattern', '') for p in algorithmic_patterns]

                                all_detected = concept_labels + pattern_labels

                                matches = 0
                                for expected in expected_concepts:
                                    total_concept_tests += 1
                                    if any(expected in detected.lower() for detected in all_detected):
                                        matches += 1
                                        concept_success += 1

                                print(f"   🧠 {entity_name}: {matches}/{len(expected_concepts)} concepts détectés")
                                print(f"      Concepts: {concept_labels[:3]}")
                                print(f"      Patterns: {pattern_labels[:2]}")
                            else:
                                print(f"   ❌ {entity_name}: erreur analyse sémantique")

                        except Exception as e:
                            print(f"   ❌ {entity_name}: exception {str(e)[:30]}...")

                    concept_rate = (concept_success / total_concept_tests * 100) if total_concept_tests > 0 else 0
                    print(
                        f"   📈 Taux de détection concepts: {concept_rate:.1f}% ({concept_success}/{total_concept_tests})")

                    print("\n6. ⚡ Test performance et cache...")

                    # Test performance avec entité complexe
                    test_entity = 'simulation'

                    # Premier appel (sans cache)
                    start_time = asyncio.get_event_loop().time()
                    context1 = await orchestrator.get_context_for_agent(
                        test_entity, 'developer', 'code_understanding', 4000
                    )
                    first_time = asyncio.get_event_loop().time() - start_time

                    # Deuxième appel (avec cache)
                    start_time = asyncio.get_event_loop().time()
                    context2 = await orchestrator.get_context_for_agent(
                        test_entity, 'developer', 'code_understanding', 4000
                    )
                    second_time = asyncio.get_event_loop().time() - start_time

                    # Vérifier que les résultats sont identiques
                    tokens_match = context1['total_tokens'] == context2['total_tokens']
                    contexts_match = list(context1['contexts'].keys()) == list(context2['contexts'].keys())

                    speedup = first_time / second_time if second_time > 0 else 1

                    print(f"   ⏱️ Première exécution: {first_time:.3f}s ({context1['total_tokens']} tokens)")
                    print(f"   ⚡ Deuxième exécution: {second_time:.3f}s ({context2['total_tokens']} tokens)")
                    print(f"   📈 Accélération: {speedup:.1f}x")
                    print(f"   ✅ Cohérence: tokens={tokens_match}, contextes={contexts_match}")

                    # Stats de cache
                    cache_stats = orchestrator.get_cache_stats()
                    total_entries = sum(getattr(stats, 'entries', 0) for stats in cache_stats.values())
                    total_hits = sum(getattr(stats, 'hits', 0) for stats in cache_stats.values())
                    total_requests = sum(
                        getattr(stats, 'hits', 0) + getattr(stats, 'misses', 0) for stats in cache_stats.values())

                    hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0

                    print(f"   💾 Cache: {total_entries} entrées, {hit_rate:.1f}% hit rate")

                    print("\n7. 🔍 Test recherche et résolution d'entités...")

                    # Test recherche fuzzy
                    search_tests = [
                        'molecular',  # Devrait trouver molecular_dynamics_main
                        'force',  # Devrait trouver forces, lennard_jones_force
                        'temp',  # Devrait trouver compute_temperature
                        'verlet'  # Devrait trouver velocity_verlet_step
                    ]

                    search_success = 0

                    for query in search_tests:
                        try:
                            results = await orchestrator.search_entities(query)

                            if results:
                                search_success += 1
                                print(f"   🔍 '{query}': {len(results)} résultats")
                                for result in results[:2]:
                                    print(f"      - {result['name']} ({result['type']}) [{result['match_type']}]")
                            else:
                                print(f"   ❌ '{query}': aucun résultat")

                        except Exception as e:
                            print(f"   ❌ '{query}': exception {str(e)[:30]}...")

                    print(f"   📈 Recherches réussies: {search_success}/{len(search_tests)}")

                    print("\n8. 🎯 Test cas d'usage complexes...")

                    # Scénario 1: Développeur débugant un problème de performance
                    print("   Scénario 1: Debug performance sur lennard_jones_force")
                    try:
                        debug_context = await orchestrator.get_context_for_agent(
                            'lennard_jones_force', 'developer', 'optimization', 4000
                        )

                        if 'error' not in debug_context.get('contexts', {}).get('local', {}):
                            insights = debug_context.get('key_insights', [])
                            recs = debug_context.get('recommendations', [])
                            patterns = debug_context.get('contexts', {}).get('semantic', {}).get('algorithmic_patterns',
                                                                                                 [])

                            print(
                                f"      ✅ {len(insights)} insights, {len(recs)} recommendations, {len(patterns)} patterns")

                            # Vérifier la présence d'insights de performance
                            perf_insights = [i for i in insights if
                                             'performance' in i.lower() or 'optimization' in i.lower()]
                            if perf_insights:
                                print(f"      🎯 Insights performance détectés: {len(perf_insights)}")
                        else:
                            print("      ❌ Erreur contexte local")

                    except Exception as e:
                        print(f"      ❌ Exception: {str(e)[:50]}...")

                    # Scénario 2: Reviewer analysant l'architecture
                    print("   Scénario 2: Review architecture du module simulation")
                    try:
                        review_context = await orchestrator.get_context_for_agent(
                            'simulation', 'reviewer', 'architecture_review', 5000
                        )

                        global_ctx = review_context.get('contexts', {}).get('global', {})
                        if 'error' not in global_ctx:
                            impact = global_ctx.get('impact_analysis', {})
                            hierarchy = global_ctx.get('module_hierarchy', {})

                            risk_level = impact.get('risk_level', 'unknown')
                            affected_modules = len(impact.get('affected_modules', []))
                            circular_deps = len(hierarchy.get('circular_dependencies', []))

                            print(f"      ✅ Risque: {risk_level}, {affected_modules} modules affectés")
                            print(f"      🔄 Dépendances circulaires: {circular_deps}")

                            agent_insights = review_context.get('agent_specific_insights', [])
                            if agent_insights:
                                print(f"      👀 Insights reviewer: {len(agent_insights)}")
                        else:
                            print("      ❌ Erreur contexte global")

                    except Exception as e:
                        print(f"      ❌ Exception: {str(e)[:50]}...")

                    # Scénario 3: Documenteur créant la documentation API
                    print("   Scénario 3: Documentation API du module forces")
                    try:
                        doc_context = await orchestrator.get_context_for_agent(
                            'forces', 'documenter', 'api_documentation', 4000
                        )

                        local_ctx = doc_context.get('contexts', {}).get('local', {})
                        if 'error' not in local_ctx:
                            main_def = local_ctx.get('main_definition', {})
                            dependencies = local_ctx.get('immediate_dependencies', [])
                            called_functions = local_ctx.get('called_functions', [])

                            signature = main_def.get('signature', '')
                            concepts = main_def.get('concepts', [])

                            print(
                                f"      ✅ Signature: {'trouvée' if signature != 'Signature not found' else 'manquante'}")
                            print(f"      📋 {len(dependencies)} dépendances, {len(called_functions)} appels")
                            print(f"      🏷️ {len(concepts)} concepts détectés")
                        else:
                            print("      ❌ Erreur contexte local")

                    except Exception as e:
                        print(f"      ❌ Exception: {str(e)[:50]}...")

                    print("\n9. 📈 Calcul des métriques de succès globales...")

                    # Calculer le taux de succès global
                    total_tests = 0
                    successful_tests = 0

                    # Tests d'entités par agent
                    for entity_results in test_results.values():
                        for test_name, result in entity_results.items():
                            if 'success' in result:
                                total_tests += 1
                                if result['success']:
                                    successful_tests += 1

                    # Ajouter les autres tests
                    total_tests += len(search_tests)
                    successful_tests += search_success

                    total_tests += total_concept_tests
                    successful_tests += concept_success

                    success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0

                    print(f"   📊 Tests réussis: {successful_tests}/{total_tests} ({success_rate:.1f}%)")
                    print(f"   ⚡ Performance cache: {speedup:.1f}x accélération")
                    print(f"   💾 Efficacité cache: {hit_rate:.1f}% hit rate")

                    print("\n10. 🧹 Nettoyage et rapport final...")

                    # Nettoyage des caches
                    await orchestrator.clear_caches()

                    # Statistiques finales
                    final_stats = orchestrator.get_index_stats()
    except Exception as e:
        print(f"      ❌ Exception: {str(e)[:50]}...")

if __name__ == "__main__":
    success = asyncio.run(test_complete_multifile_system())
    print(f"\n{'✅ SUCCÈS' if success else '❌ ÉCHEC'}")