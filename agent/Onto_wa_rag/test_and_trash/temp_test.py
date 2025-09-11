"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

import logging
from fparser.common.readfortran import FortranStringReader
from fparser.two.parser import ParserFactory
from fparser.two.utils import walk

# Configurez un logger simple pour voir les messages d'erreur de fparser
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger()

# Votre code Fortran exact
fortran_code = """
module constants_types
    use iso_fortran_env, only: real64
    implicit none
    
    ! Physical constants
    real(real64), parameter :: kb = 1.380649e-23_real64  ! Boltzmann constant
    real(real64), parameter :: na = 6.02214076e23_real64 ! Avogadro number
    real(real64), parameter :: pi = 3.141592653589793_real64
    
    ! Simulation parameters
    integer, parameter :: max_particles = 10000
    real(real64), parameter :: default_dt = 0.001_real64
    
    ! Derived types
    type :: particle_t
        real(real64) :: x, y, z           ! position
        real(real64) :: vx, vy, vz        ! velocity  
        real(real64) :: fx, fy, fz        ! force
        real(real64) :: mass
        integer :: id
    end type particle_t
    
    type :: system_t
        type(particle_t), allocatable :: particles(:)
        integer :: n_particles
        real(real64) :: box_size
        real(real64) :: temperature
        real(real64) :: total_energy
    end type system_t
    
end module constants_types 

"""

print("--- DEBUT DU PARSING FPARSER BRUT ---")
node_count = 0
try:
    reader = FortranStringReader(fortran_code, ignore_comments=False)
    parser = ParserFactory().create(std="f2008")
    ast = parser(reader)

    if ast:
        print("AST créé avec succès. Parcours des noeuds...")
        for node in walk(ast):
            node_count += 1
            # Affiche la représentation de chaque noeud
            print(f"  - Node {node_count}: {repr(node)}")
        print(f"\n--- PARSING TERMINÉ. Total de {node_count} noeuds trouvés. ---")
    else:
        print("--- ERREUR: L'AST retourné par fparser est None! ---")

except Exception:
    log.error("--- ERREUR CATASTROPHIQUE DURANT LE PARSING FPARSER ---", exc_info=True)