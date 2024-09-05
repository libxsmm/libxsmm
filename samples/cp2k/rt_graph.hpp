/*
 * Copyright (c) 2019 Simon Frasch
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef RT_GRAPH_HPP_GUARD
#define RT_GRAPH_HPP_GUARD

#include <atomic>
#include <chrono>
#include <cstddef>
#include <deque>
#include <list>
#include <string>
#include <vector>

namespace rt_graph {

using ClockType = std::chrono::high_resolution_clock;

// Selection of available statistics
enum class Stat {
  Count,            // Number of measurements
  Total,            // Total accumulated time
  Mean,             // Mean time
  Median,           // Median time
  QuartileHigh,     // Third quartile time
  QuartileLow,      // First quartile time
  Min,              // Mininum time
  Max,              // Maximum time
  Percentage,       // Percentage of accumulated time with respect to the top-level node in graph
  ParentPercentage  // Percentage of accumulated time with respect to the parent node in graph
};

// internal helper functionality
namespace internal {

enum class TimeStampType { Start, Stop, Empty };

struct TimeStamp {
  TimeStamp() : type(TimeStampType::Empty) {}

  // Identifier pointer must point to compile time string literal
  TimeStamp(const char* identifier, const TimeStampType& stampType)
      : time(ClockType::now()), identifierPtr(identifier), type(stampType) {}

  ClockType::time_point time;
  const char* identifierPtr;
  TimeStampType type;
};

struct TimingNode {
  std::string identifier;
  std::vector<double> timings;
  std::list<TimingNode> subNodes;
};
}  // namespace internal

// Processed timings results.
class TimingResult {
public:
  TimingResult(std::list<internal::TimingNode> rootNodes, std::string warnings)
      : rootNodes_(std::move(rootNodes)), warnings_(std::move(warnings)) {}

  // Get json representation of the full graph with all timings. Unit of time is seconds.
  auto json() const -> std::string;

  // Get all timings for given identifier
  auto get_timings(const std::string& identifier) const -> std::vector<double>;

  // Print graph statistic to string.
  auto print(std::vector<Stat> statistic = {Stat::Count, Stat::Total, Stat::Percentage,
                                            Stat::ParentPercentage, Stat::Median, Stat::Min,
                                            Stat::Max}) const -> std::string;

private:
  std::list<internal::TimingNode> rootNodes_;
  std::string warnings_;
};

class ScopedTiming;

// Timer class, which allows to start / stop measurements with a given identifier.
class Timer {
public:
  // reserve space for 1000'000 measurements
  Timer() { timeStamps_.reserve(2 * 1000 * 1000); }

  // reserve space for given number of measurements
  explicit Timer(std::size_t reserveCount) { timeStamps_.reserve(2 * reserveCount); }

  // start with string literal identifier
  template <std::size_t N>
  inline auto start(const char (&identifierPtr)[N]) -> void {
    atomic_signal_fence(std::memory_order_seq_cst);  // only prevents compiler reordering
    timeStamps_.emplace_back(identifierPtr, internal::TimeStampType::Start);
    atomic_signal_fence(std::memory_order_seq_cst);  // only prevents compiler reordering
  }

  // start with string identifier (storing string object comes with some additional overhead)
  inline auto start(std::string identifier) -> void {
    atomic_signal_fence(std::memory_order_seq_cst);  // only prevents compiler reordering
    identifierStrings_.emplace_back(std::move(identifier));
    timeStamps_.emplace_back(identifierStrings_.back().c_str(), internal::TimeStampType::Start);
    atomic_signal_fence(std::memory_order_seq_cst);  // only prevents compiler reordering
  }

  // stop with string literal identifier
  template <std::size_t N>
  inline auto stop(const char (&identifierPtr)[N]) -> void {
    atomic_signal_fence(std::memory_order_seq_cst);  // only prevents compiler reordering
    timeStamps_.emplace_back(identifierPtr, internal::TimeStampType::Stop);
    atomic_signal_fence(std::memory_order_seq_cst);  // only prevents compiler reordering
  }

  // stop with string identifier (storing string object comes with some additional overhead)
  inline auto stop(std::string identifier) -> void {
    atomic_signal_fence(std::memory_order_seq_cst);  // only prevents compiler reordering
    identifierStrings_.emplace_back(std::move(identifier));
    timeStamps_.emplace_back(identifierStrings_.back().c_str(), internal::TimeStampType::Stop);
    atomic_signal_fence(std::memory_order_seq_cst);  // only prevents compiler reordering
  }

  // clear timer and reserve space for given number of new measurements.
  inline auto clear(std::size_t reserveCount) -> void {
    timeStamps_.clear();
    identifierStrings_.clear();
    this->reserve(reserveCount);
  }

  // reserve space for given number of measurements. Can prevent allocations at start / stop calls.
  inline auto reserve(std::size_t reserveCount) -> void { timeStamps_.reserve(reserveCount); }

  // process timings into result type
  auto process() const -> TimingResult;

private:
  inline auto stop_with_ptr(const char* identifierPtr) -> void {
    atomic_signal_fence(std::memory_order_seq_cst);  // only prevents compiler reordering
    timeStamps_.emplace_back(identifierPtr, internal::TimeStampType::Stop);
    atomic_signal_fence(std::memory_order_seq_cst);  // only prevents compiler reordering
  }

  friend ScopedTiming;

  std::vector<internal::TimeStamp> timeStamps_;
  std::deque<std::string>
      identifierStrings_;  // pointer to elements always remain valid after push back
};

// Helper class, which calls start() upon creation and stop() on timer when leaving scope with given
// identifier.
class ScopedTiming {
public:
  // timer reference must be valid for the entire lifetime
  template <std::size_t N>
  ScopedTiming(const char (&identifierPtr)[N], Timer& timer)
      : identifierPtr_(identifierPtr), timer_(timer) {
    timer_.start(identifierPtr);
  }

  ScopedTiming(std::string identifier, Timer& timer)
      : identifierPtr_(nullptr), identifier_(std::move(identifier)), timer_(timer) {
    timer_.start(identifier_);
  }

  ScopedTiming(const ScopedTiming&) = delete;
  ScopedTiming(ScopedTiming&&) = delete;
  auto operator=(const ScopedTiming&) -> ScopedTiming& = delete;
  auto operator=(ScopedTiming &&) -> ScopedTiming& = delete;

  ~ScopedTiming() {
    if (identifierPtr_) {
      timer_.stop_with_ptr(identifierPtr_);
    } else {
      timer_.stop(std::move(identifier_));
    }
  }

private:
  const char* identifierPtr_;
  std::string identifier_;
  Timer& timer_;
};

}  // namespace rt_graph

#endif
