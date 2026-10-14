// Banked on-chip SRAM TLM target (design note §3.5). Accepts every request
// at its sample point; bank conflicts show up as later response cycles.
#pragma once
#include "mem/sampled_target.h"
#include "mem/sram_timing.h"

namespace apm {

class Sram : public SampledTarget {
 public:
  Sram(sc_core::sc_module_name name, const SramParams& p);

  const SramTiming& timing() const { return timing_; }
  void report(std::map<std::string, std::uint64_t>& out) const override;

 protected:
  bool accept(const tlm::tlm_generic_payload& trans, Cycle accept, Cycle& response) override;

 private:
  SramTiming timing_;
};

}  // namespace apm
