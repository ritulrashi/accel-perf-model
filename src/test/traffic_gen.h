// Test-only scripted initiator (design note §3.7). Lets a directed test drive
// one target without DMA timing mixed in. Each request is sent at
// max(scripted cycle, first cycle the socket is free); several may be
// outstanding.
#pragma once
#include <systemc>
#include <tlm>
#include <tlm_utils/simple_initiator_socket.h>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/cycle.h"

namespace apm {

struct TgRequest {
  Cycle at;
  bool write;
  std::uint64_t addr;
  std::uint32_t size;
};

struct TgResult {
  Cycle issued = 0, response = 0;
  bool done = false;
};

class TrafficGen : public sc_core::sc_module {
 public:
  tlm_utils::simple_initiator_socket<TrafficGen> socket;

  TrafficGen(sc_core::sc_module_name name, std::vector<TgRequest> script);

  const std::vector<TgResult>& results() const { return results_; }
  void report(std::map<std::string, std::uint64_t>& out) const;

 private:
  void act();
  tlm::tlm_sync_enum nb_transport_bw(tlm::tlm_generic_payload& trans, tlm::tlm_phase& phase,
                                     sc_core::sc_time& delay);

  std::vector<TgRequest> script_;
  std::vector<std::vector<std::uint8_t>> data_;
  std::vector<std::unique_ptr<tlm::tlm_generic_payload>> payloads_;  // not copyable
  std::unordered_map<tlm::tlm_generic_payload*, std::size_t> index_of_;
  std::vector<TgResult> results_;
  std::size_t next_ = 0;
  bool busy_ = false;  // waiting for END_REQ
  sc_core::sc_event act_next_;
};

}  // namespace apm
