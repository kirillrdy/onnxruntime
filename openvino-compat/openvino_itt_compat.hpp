#pragma once

// Zig gives every source in a module one unified include search path. Some
// OpenVINO sources include the basename `itt.hpp`, which can consequently
// select another component's header. Load the core header explicitly, then
// disable profiling scopes that depend on CMake-generated ITT declarations.
#include "src/core/src/itt.hpp"

#undef OV_OP_SCOPE
#define OV_OP_SCOPE(region)

#undef OV_ITT_SCOPED_REGION_BASE
#define OV_ITT_SCOPED_REGION_BASE(...)
