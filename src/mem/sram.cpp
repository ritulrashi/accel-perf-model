#include "mem/sram.h"

namespace apm {

Sram::Sram(sc_core::sc_module_name name, const SramParams& p)
    : SampledTarget(name, p.capacity_bytes), timing_(p) {}

bool Sram::accept(const tlm::tlm_generic_payload& trans, Cycle accept, Cycle& response) {
  response = timing_.schedule(accept, trans.get_address(), trans.get_data_length()).response;
  return true;
}

void Sram::report(std::map<std::string, std::uint64_t>& out) const {
  const std::string p = std::string(basename()) + ".";
  out[p + "lines_accessed"] = timing_.lines_accessed();
  out[p + "conflict_line_cycles"] = timing_.conflict_line_cycles();
}

}  // namespace apm
