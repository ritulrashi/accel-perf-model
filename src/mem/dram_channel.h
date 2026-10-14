// DRAM channel TLM target (design note §3.4): DramTiming formulas plus a
// request queue of depth Q that provides backpressure through END_REQ.
#pragma once
#include <functional>
#include <queue>
#include <vector>

#include "mem/dram_timing.h"
#include "mem/sampled_target.h"

namespace apm {

class DramChannel : public SampledTarget {
 public:
  DramChannel(sc_core::sc_module_name name, const DramParams& p);

  const DramTiming& timing() const { return timing_; }
  void report(std::map<std::string, std::uint64_t>& out) const override;

 protected:
  void begin_sample(Cycle c) override;
  bool accept(const tlm::tlm_generic_payload& trans, Cycle accept, Cycle& response) override;

 private:
  DramTiming timing_;
  // Response cycles of accepted requests still holding a queue slot.
  std::priority_queue<Cycle, std::vector<Cycle>, std::greater<Cycle>> in_queue_;
  std::uint64_t bytes_read_ = 0, bytes_written_ = 0;
};

}  // namespace apm
