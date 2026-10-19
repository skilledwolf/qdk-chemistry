Wavefunction
============

The :class:`~qdk_chemistry.data.Wavefunction` class in QDK/Chemistry represents quantum mechanical wavefunctions for molecular systems.
This class provides access to wavefunction coefficients, determinants, reduced density matrices (:term:`RDM`), orbital entropies :cite:`Boguslawski2015`, and other quantum chemical properties.

Overview
--------

A wavefunction in quantum chemistry describes the quantum state of a molecular system.
In QDK/Chemistry, the :class:`~qdk_chemistry.data.Wavefunction` class encapsulates various wavefunction types, from simple single-determinant Hartree-Fock wavefunctions to complex multi-reference wavefunctions.

The class uses a container-based design where different wavefunction types (Slater determinants, configuration interaction, coupled cluster, etc.) are implemented as specialized container classes, while the main :class:`~qdk_chemistry.data.Wavefunction` class provides a unified interface.

Mathematical representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Wavefunctions are represented as linear combinations of determinants:

.. math::

   |\Psi\rangle = \sum_I c_I |\Phi_I\rangle

where :math:`c_I` are expansion coefficients and :math:`|\Phi_I\rangle` are Slater determinants.

For post-Hartree-Fock methods like coupled cluster, the wavefunction is expressed in terms of cluster operators:

.. math::

   |\Psi_{CC}\rangle = e^{\hat{T}} |\Phi_0\rangle

where :math:`\hat{T} = \hat{T}_1 + \hat{T}_2 + ...` is the cluster operator and :math:`|\Phi_0\rangle` is the reference determinant.


Container types
---------------

QDK/Chemistry supports different wavefunction container types for various quantum chemistry methods:

Slater determinant container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Single-determinant wavefunctions (e.g., from Hartree-Fock calculations).

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/wavefunction_container.py
      :language: python
      :start-after: # start-cell-create-slater
      :end-before: # end-cell-create-slater

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/wavefunction_container.cpp
      :language: cpp
      :start-after: // start-cell-create-slater
      :end-before: // end-cell-create-slater

SCI wavefunction container
~~~~~~~~~~~~~~~~~~~~~~~~~~

Sparse multi-determinant wavefunctions for Selected Configuration Interaction methods.

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/wavefunction_container.py
      :language: python
      :start-after: # start-cell-create-sci
      :end-before: # end-cell-create-sci

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/wavefunction_container.cpp
      :language: cpp
      :start-after: // start-cell-create-sci
      :end-before: // end-cell-create-sci

CAS wavefunction container
~~~~~~~~~~~~~~~~~~~~~~~~~~

A multi-determinant wavefunction from Complete Active Space methods (:term:`CASSCF`/:term:`CASCI`).

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/wavefunction_container.py
      :language: python
      :start-after: # start-cell-create-cas
      :end-before: # end-cell-create-cas

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/wavefunction_container.cpp
      :language: cpp
      :start-after: // start-cell-create-cas
      :end-before: // end-cell-create-cas

Amplitude wavefunction container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A single container, :class:`~qdk_chemistry.data.AmplitudeContainer`, represents amplitude-based correlated wavefunctions such as :term:`MP2` and coupled cluster.
It stores a reference wavefunction together with T1/T2 excitation amplitudes, and records which correlated method produced them via :class:`~qdk_chemistry.data.AmplitudeType` (``MP2``, ``CCSD``, or ``Unspecified``).
It is pure storage: the amplitudes are supplied by the producing algorithm and are not expanded into a determinant/coefficient representation, so determinant-, coefficient-, and :term:`RDM`-based accessors raise.

The following example tags the amplitudes as :term:`MP2` (T1 is zero for MP2):

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/wavefunction_container.py
      :language: python
      :start-after: # start-cell-create-mp2
      :end-before: # end-cell-create-mp2

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/wavefunction_container.cpp
      :language: cpp
      :start-after: // start-cell-create-mp2
      :end-before: // end-cell-create-mp2

The same container holds coupled cluster amplitudes when tagged as ``CCSD``:

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/wavefunction_container.py
      :language: python
      :start-after: # start-cell-create-cc
      :end-before: # end-cell-create-cc

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/wavefunction_container.cpp
      :language: cpp
      :start-after: // start-cell-create-cc
      :end-before: // end-cell-create-cc

Matrix product state container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~qdk_chemistry.data.MPSContainer` stores an open-boundary matrix product state supplied by an external producer. Each :class:`~qdk_chemistry.data.MPSSite` has tensor axes ``(left bond, physical state, right bond)``; its dense export is packed as a matrix with row ``left * physical_dimension + physical`` and column ``right``. Construction and serialization preserve the supplied coefficients, including their scale and complex phase; they do not normalize or canonicalize the state.

Supplied RDMs describe normalized expectation values, independently of the scale of the stored tensors. Each 1-RDM must have shape ``(n,n)`` and each flattened 2-RDM must contain ``n**4`` finite entries, where ``n`` is the number of active sites. Their declared index spaces, block labels and dimensions are validated on construction and after JSON/HDF5 deserialization. These structural checks do not establish physical consistency between the supplied RDMs and tensors.

The ``active_one_rdm`` and ``active_two_rdm`` arguments accept real or complex ``SymmetryBlockedTensor`` objects. They preserve explicitly declared block equivalence, including through JSON/HDF5 serialization. Equivalence must belong to the supplied RDM itself; shared orbital coefficients or equal particle counts do not establish it. For example, a producer whose two-orbital state has equal alpha and beta 1-RDM blocks can declare that explicitly when it builds the container from its ``sites``:

.. code-block:: python

   import numpy as np
   from qdk_chemistry.data import ModelOrbitals, MPSContainer, Wavefunction
   from qdk_chemistry.data import symmetry as sym

   equivalent = sym.SymmetryProduct([sym.axes.spin(1, True)])
   alpha = sym.SymmetryLabel([sym.axes.alpha()])
   beta = sym.SymmetryLabel([sym.axes.beta()])
   one_rdm = sym.SymmetryBlockedTensorRank2(
       [equivalent, equivalent], [{alpha: 2, beta: 2}] * 2,
       [((alpha, alpha), 0.5 * np.eye(2))],
   )
   wavefunction = Wavefunction(MPSContainer(sites, ModelOrbitals(2), active_one_rdm=one_rdm))

The current RDM interface supports spin-diagonal blocks. An explicit equivalent 1-RDM needs one representative for ``aa`` and ``bb``. An equivalent 2-RDM still needs a supplied representative for each block orbit: ``aaaa``/``bbbb`` and ``aabb``/``bbaa``. Declaring equivalence within either orbit says nothing about the other orbit. Unsupported block patterns are rejected. Missing independent blocks are errors, while explicitly supplied zero blocks are valid.

A four-block 2-RDM may also supply ``bbaa`` independently of ``aabb``; tracing uses each supplied block directly. Without an explicit fourth block, the existing three-block convention exchanges the index pairs of ``aabb``. Use ``active_two_rdm()`` to access all four blocks; ``get_active_two_rdm_spin_dependent()`` returns three blocks.

The per-channel array arguments remain available: supply both 1-RDM arrays or all three 2-RDM arrays. Their omission never declares equivalence. Do not combine arrays with a blocked tensor for the same RDM rank. A traced RDM may be supplied on its own; it does not imply a decomposition into channels.

Natural occupations are descending eigenvalues of supplied Hermitian 1-RDMs; they require no tensor contraction. A spin basis requires alpha and beta blocks supplied directly or through declared equivalence, diagonalized block by block. A spin-free basis uses the spin-traced 1-RDM. Equal particle counts alone do not justify splitting a spin-traced RDM.

Producers may also supply single-orbital entropies and mutual information in active-orbital order. They are serialized with the container and feed entropy-based active-space selection without requiring the full two-particle RDM. Two-orbital entropies may be supplied with single-orbital entropies instead; the mutual information they imply is derived on construction.

The ``provenance`` argument takes a JSON object describing how the state was produced. It is stored and serialized verbatim and is not interpreted. A DMRG producer might record its name and version, the variational energy, the discarded weight of the final sweep, the bond dimensions, and the content hash of the Hamiltonian it optimized; the discarded weight in particular is needed to judge or extrapolate a truncated state.

Because supplied RDMs and entropies cannot be regenerated from the sites, they are part of the container's content hash. Values that getters derive from other supplied data are excluded, so the hash does not depend on which getters have run: a traced RDM supplied next to its blocked form, single-orbital entropies supplied next to both blocked RDMs when those are real, and two-orbital entropies. Such values must be consistent with the data they derive from. The provenance describes how the state was produced rather than the state itself, so it is not part of the hash: the same state recorded by two runs has one hash.

Fermionic coefficient convention
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The default four-state physical basis ``(0,u,d,2)`` denotes

.. math::

   |0\rangle = |\mathrm{vac}\rangle,\qquad
   |u\rangle = a^\dagger_\alpha |\mathrm{vac}\rangle,\qquad
   |d\rangle = a^\dagger_\beta |\mathrm{vac}\rangle,\qquad
   |2\rangle = a^\dagger_\alpha a^\dagger_\beta |\mathrm{vac}\rangle.

Contracting the sites gives coefficients in the product basis whose creation operators are written from left to right in chain-site order, alpha before beta within each site. An explicit ``physical_basis`` may reorder or restrict the local states; it does not change their phases. The binary basis ``(0,1)`` similarly denotes the vacuum and one occupied spinless mode. Producers whose local states use another operator order convert with a site-local sign on the affected physical states; for example, a doubly occupied state defined as :math:`a^\dagger_\beta a^\dagger_\alpha|\mathrm{vac}\rangle` is the negative of ``2``.

``site_to_orbital_order[s]`` identifies the active-orbital slot represented by chain site ``s``; an omitted order is the identity. These are active-space slots, not indices in the full molecular-orbital basis. The container stores the mapping without rearranging tensors or changing signs.

Determinant coefficients, as stored in :class:`~qdk_chemistry.data.StateVectorContainer`, order the creation operators with all alpha modes before all beta modes, each in ascending orbital order. :meth:`~qdk_chemistry.data.MPSContainer.get_coefficient` contracts the sites for one determinant and multiplies by the parity of the permutation that sorts the chain-ordered operators into that order. The sign is required even for the identity site order: ``|2u>`` creates :math:`a^\dagger_{0\alpha}a^\dagger_{0\beta}a^\dagger_{1\alpha}`, which needs one exchange to reach determinant order. Spinless modes follow the same rule when their site order is permuted.

A small reference state is

.. math::

   |\Psi\rangle = \tfrac12\left(|2u0\rangle + i|u02\rangle - |udu\rangle - i|02u\rangle\right)

in chain order. For ``site_to_orbital_order = [2,0,1]`` its nonzero determinant coefficients are listed below, with occupations written as ``alpha_0 alpha_1 alpha_2 beta_0 beta_1 beta_2``.

.. list-table:: Signed reference amplitudes
   :header-rows: 1

   * - Chain configuration
     - Determinant occupation bits
     - Coefficient
   * - ``2u0``
     - ``101001``
     - :math:`1/2`
   * - ``u02``
     - ``011010``
     - :math:`-i/2`
   * - ``udu``
     - ``011100``
     - :math:`-1/2`
   * - ``02u``
     - ``110100``
     - :math:`i/2`

``python/tests/test_mps_conventions.py`` checks these amplitudes before and after JSON and HDF5 round trips, and checks an exact MPS of a ground state against the signed determinant coefficients computed by MACIS.

Properties
~~~~~~~~~~

The :class:`~qdk_chemistry.data.Wavefunction` class provides access to various quantum chemical properties. Availability depends on the specific container type:


.. list-table:: Property availability by container type
   :header-rows: 1
   :widths: 30 15 15 15 15 15

   * - Property
     - Slater determinant
     - :term:`CAS`
     - :term:`SCI`
     - :term:`MP2`
     - Coupled cluster

   * - **Coefficients**
     - ✓
     - ✓
     - ✓
     - ✗
     - ✗

   * - **Determinants**
     - ✓
     - ✓
     - ✓
     - ✗
     - ✗

   * - **Electron counts**
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓

   * - **Orbital occupations**
     - ✓
     - ✓
     - ✓
     - ✗
     - ✓

   * - **1-RDMs (spin-dependent)**
     - ✓
     - ✓
     - ✓
     - ✗
     - ✓

   * - **1-RDMs (spin-traced)**
     - ✓
     - ✓
     - ✓
     - ✗
     - ✓

   * - **2-RDMs (spin-dependent)**
     - ✓
     - ✓*
     - ✓*
     - ✗
     - ✓

   * - **2-RDMs (spin-traced)**
     - ✓
     - ✓*
     - ✓*
     - ✗
     - ✓

   * - **Orbital entropies**
     - ✓
     - ✓*
     - ✓*
     - ✗
     - ✓

   * - **T1/T2 amplitudes**
     - ✗
     - ✗
     - ✗
     - ✓
     - ✓

   * - **Overlap calculations**
     - ✗
     - ✓
     - ✗
     - ✗
     - ✗

   * - **Norm calculations**
     - ✓
     - ✓
     - ✗
     - ✗
     - ✗




Legend:

- ✓ Available and implemented

- ✗ Not available (method not implemented)

- ✓* Implemented and available only if 2-RDMs were provided during construction

Accessing wavefunction data
---------------------------

The :class:`~qdk_chemistry.data.Wavefunction` class provides methods to access coefficients, determinants, and derived properties:

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/wavefunction_container.py
      :language: python
      :start-after: # start-cell-access-data
      :end-before: # end-cell-access-data

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/wavefunction_container.cpp
      :language: cpp
      :start-after: // start-cell-access-data
      :end-before: // end-cell-access-data

Total spin
-----------

The :meth:`~qdk_chemistry.data.Wavefunction.compute_s_squared` method evaluates
the total-spin expectation value :math:`\langle S^2\rangle` from the real,
spin-resolved active-space one- and two-particle :term:`RDM` blocks. For a pure
spin eigenstate with spin quantum number :math:`S`, the result is
:math:`S(S+1)`.

The calculation requires alpha and beta molecular orbitals to share a common
spatial basis. Restricted closed-shell and restricted open-shell wavefunctions
meet this requirement; unrestricted orbitals are rejected because their
spin-flip contractions additionally require the alpha--beta molecular-orbital
overlap. The method also raises if the required RDM blocks are unavailable,
their symmetry-blocked extents are inconsistent, or they are complex-valued.

.. tab:: Python API

   .. code-block:: python

      s_squared = wavefunction.compute_s_squared()

.. tab:: C++ API

   .. code-block:: cpp

      double s_squared = wavefunction->compute_s_squared();

Accessing cluster amplitudes
----------------------------

For :term:`MP2` and coupled cluster wavefunctions, one can access T1 and T2 cluster amplitudes.

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/wavefunction_container.py
      :language: python
      :start-after: # start-cell-access-amplitudes
      :end-before: # end-cell-access-amplitudes

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/wavefunction_container.cpp
      :language: cpp
      :start-after: // start-cell-access-amplitudes
      :end-before: // end-cell-access-amplitudes

Related classes
---------------

- :doc:`Serialization <serialization>`: Data serialization and deserialization
- :doc:`Settings <../algorithms/settings>`: Configuration settings for algorithms
- :doc:`Active space methods <../algorithms/active_space>`: Active space selection from wavefunctions
- :doc:`Structure <structure>`: Molecular structure that defines the system
- :doc:`Orbitals <orbitals>`: Orbital basis set for the wavefunction
- :doc:`Hamiltonian <hamiltonian>`: Electronic Hamiltonian constructed from wavefunction
- :doc:`ScfSolver <../algorithms/scf_solver>`: Algorithm that produces :term:`SCF` wavefunctions
- :doc:`MCCalculator <../algorithms/mc_calculator>`: Algorithm for multi-configuration wavefunctions

Further reading
---------------

- :doc:`Settings <../algorithms/settings>`: Configuration settings for algorithms
- :doc:`Serialization <../data/serialization>`: Data serialization and deserialization
- :doc:`Active space methods <../algorithms/active_space>`: Active space selection from wavefunctions
