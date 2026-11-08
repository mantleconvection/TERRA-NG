# Flag fields and masks {#flag-fields-and-masks}

To mark grid nodes with certain properties, flag fields are used in various places in the code.
A flag field is just a grid of integers where the integers represent some properties of a node.

Mostly flag fields are used for two purposes:
* to indicate whether a node is duplicated
* to indicate whether a node is a boundary node (and to indicate the boundary type of the associated degree of freedom)

Flag fields are just standard grids (see the [Grids Section](02_grid_subdomains.md) for details) with an enum class as their value type. Using enum classes is type safe as flag fields with different value types cannot be mixed.

The value type should implement the \ref terra::util::FlagLike concept. It is basically an enum class with a special
entry `NO_FLAG`, which is used to indicate when no flag is set. That just ensures that operations on flag 
fields are well-defined (and there are no subtle errors when using bitwise operations on fields that are zero). 
Use \ref terra::util::has_flag() to check whether a flag field has a certain flag set.

### Node ownership

Since nodes are overlapping at subdomain boundaries, communication and reduction operations must be carefully executed
with duplicated nodes in mind. The \ref terra::grid::NodeOwnershipFlag type is used to mark nodes as either `OWNED` or
not (equals `NO_FLAG`). For each set of duplicated nodes, exactly one node is marked as `OWNED` globally. All non-duplicated nodes (e.g., 
in the interior of a subdomain) are also marked as `OWNED`. This way, a reduction operation can be executed by just 
ignoring all nodes that are not marked as `OWNED`.

The corresponding flag field can be set up via a call to \ref terra::grid::setup_node_ownership_mask_data().
An instance of such a flag field must be passed to the constructor of many classes that need to know which nodes are
owned by the current subdomain (particularly finite element functions / coefficient vectors like 
\ref terra::linalg::VectorQ1Scalar).

It is good practice to call \ref terra::grid::setup_node_ownership_mask_data() once at the beginning of a simulation and
pass the same returned flag field to all classes that need to know about node ownership.

### Boundary flags

For the thick spherical shell, a boundary flag is used to indicate the type of the boundary node (see
\ref terra::grid::shell::ShellBoundaryFlag for details). The flag values are `CMB` for a node on the inner (core-mantle)
boundary, `SURFACE` for a node on the outer boundary, and `INNER` for a node in the interior, while `BOUNDARY` indicates
nodes belonging to either the inner or outer boundary.

Just like for the node ownership flag, the corresponding flag field can be set up via a call to 
\ref terra::grid::shell::setup_boundary_mask_data().
