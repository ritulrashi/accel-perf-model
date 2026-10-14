// TLM extension that lets every component append trace segments to the burst
// a payload belongs to (Phase 3 design note §3.2). Payloads without the
// extension (Phase 1/2 traffic, staging) are simply not traced.
#pragma once
#include <tlm>

#include <vector>

#include "common/trace.h"

namespace apm {

struct TraceExt : tlm::tlm_extension<TraceExt> {
  std::vector<Segment>* segs = nullptr;  // owned by the DMA's burst

  tlm::tlm_extension_base* clone() const override {
    auto* e = new TraceExt;
    e->segs = segs;
    return e;
  }
  void copy_from(const tlm::tlm_extension_base& other) override {
    segs = static_cast<const TraceExt&>(other).segs;
  }
};

inline void add_trace(tlm::tlm_generic_payload& t, Cycle from, Cycle to, StallCat cat) {
  TraceExt* e = t.get_extension<TraceExt>();
  if (e != nullptr && e->segs != nullptr && to > from) e->segs->push_back({from, to, cat});
}

}  // namespace apm
