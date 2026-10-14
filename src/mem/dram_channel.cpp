#include "mem/dram_channel.h"

namespace apm {

DramChannel::DramChannel(sc_core::sc_module_name name, const DramParams& p)
    : SampledTarget(name, p.capacity_bytes), timing_(p) {}

void DramChannel::begin_sample(Cycle c) {
  // A slot released by a response at cycle c is free at the sample point of c.
  while (!in_queue_.empty() && in_queue_.top() <= c) in_queue_.pop();
}

bool DramChannel::accept(const tlm::tlm_generic_payload& trans, Cycle accept, AcceptInfo& info) {
  if (in_queue_.size() >= timing_.params().queue_depth) return false;
  const DramSchedule s = timing_.schedule(accept, trans.get_address(), trans.get_data_length());
  info.response = s.data_end;
  info.cat = StallCat::Dram;
  in_queue_.push(info.response);
  (trans.is_read() ? bytes_read_ : bytes_written_) += trans.get_data_length();
  return true;
}

void DramChannel::report(std::map<std::string, std::uint64_t>& out) const {
  const std::string p = std::string(basename()) + ".";
  out[p + "row_hits"] = timing_.row_hits();
  out[p + "row_misses"] = timing_.row_misses();
  out[p + "bus_busy_cycles"] = timing_.bus_busy_cycles();
  out[p + "bytes_read"] = bytes_read_;
  out[p + "bytes_written"] = bytes_written_;
  out[p + "backpressure_cycles"] = backpressure_cycles();
}

}  // namespace apm
