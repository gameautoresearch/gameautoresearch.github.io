// SPDX-License-Identifier: Apache-2.0
// The task spec a native binary runs: the pinned generated fixed8 data with this build's
// match form (rules/match_form.hpp) applied. Without form flags it equals make_fixed8_spec().
#pragma once
#include "generated-v2/config.hpp"
#include "rules/match_form.hpp"
namespace clash_native {
inline NativeTaskSpec make_task_spec(){NativeTaskSpec s=make_fixed8_spec();apply_match_form(s.rules);return s;}
}
