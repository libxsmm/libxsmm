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

#include "rt_graph.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <ostream>
#include <ratio>
#include <sstream>
#include <string>
#include <tuple>

namespace rt_graph {

// ======================
// internal helper
// ======================
namespace internal {
namespace {

struct Format {
  Format(Stat stat_) : stat(stat_) {
    switch (stat_) {
      case Stat::Count:
        header = "#";
        space = 6;
        break;
      case Stat::Total:
        header = "Total";
        space = 14;
        break;
      case Stat::Mean:
        header = "Mean";
        space = 14;
        break;
      case Stat::Median:
        header = "Median";
        space = 14;
        break;
      case Stat::QuartileHigh:
        header = "Quartile High";
        space = 14;
        break;
      case Stat::QuartileLow:
        header = "Quartile Low";
        space = 14;
        break;
      case Stat::Min:
        header = "Min";
        space = 14;
        break;
      case Stat::Max:
        header = "Max";
        space = 14;
        break;
      case Stat::Percentage:
        header = "%";
        space = 11;
        break;
      case Stat::ParentPercentage:
        header = "Parent %";
        space = 11;
        break;
    }
  }

  Stat stat;
  std::string header;
  std::size_t space;
};

// format time input in seconds into string with appropriate unit
auto format_time(const double time_seconds) -> std::string {
  if (time_seconds <= 0.0) return std::string("0 s");

  // time is always greater than 0 here
  const double exponent = std::log10(std::abs(time_seconds));
  const int siExponent = static_cast<int>(std::floor(exponent / 3.0) * 3);

  std::stringstream result;
  result << std::fixed << std::setprecision(2);
  result << time_seconds * std::pow(10.0, static_cast<double>(-siExponent));
  result << " ";
  switch (siExponent) {
    case 24:
      result << "Y";
      break;
    case 21:
      result << "Z";
      break;
    case 18:
      result << "E";
      break;
    case 15:
      result << "P";
      break;
    case 12:
      result << "T";
      break;
    case 9:
      result << "G";
      break;
    case 6:
      result << "M";
      break;
    case 3:
      result << "k";
      break;
    case 0:
      break;
    case -3:
      result << "m";
      break;
    case -6:
      result << "u";
      break;
    case -9:
      result << "n";
      break;
    case -12:
      result << "p";
      break;
    case -15:
      result << "f";
      break;
    case -18:
      result << "a";
      break;
    case -21:
      result << "z";
      break;
    case -24:
      result << "y";
      break;
    default:
      result << "?";
  }
  result << "s";
  return result.str();
}

auto calc_median(const std::vector<double>::const_iterator& begin,
                 const std::vector<double>::const_iterator& end) -> double {
  const auto n = end - begin;
  if (n == 0) return 0.0;
  if (n % 2 == 0) {
    return (*(begin + n / 2) + *(begin + n / 2 - 1)) / 2.0;
  } else {
    return *(begin + n / 2);
  }
}

auto print_stat(std::ostream& out, const Format& format, const std::vector<double>& sortedTimings,
                double totalSum, double parentSum, double currentSum) -> void {
  switch (format.stat) {
    case Stat::Count:
      out << std::right << std::setw(format.space) << sortedTimings.size();
      break;
    case Stat::Total:
      out << std::right << std::setw(format.space) << format_time(currentSum);
      break;
    case Stat::Mean:
      out << std::right << std::setw(format.space)
          << format_time(currentSum / sortedTimings.size());
      break;
    case Stat::Median:
      out << std::right << std::setw(format.space)
          << format_time(calc_median(sortedTimings.begin(), sortedTimings.end()));
      break;
    case Stat::QuartileHigh: {
      const double upperQuartile =
          calc_median(sortedTimings.begin() + sortedTimings.size() / 2 +
                          (sortedTimings.size() % 2) * (sortedTimings.size() > 1),
                      sortedTimings.end());
      out << std::right << std::setw(format.space) << format_time(upperQuartile);
    } break;
    case Stat::QuartileLow: {
      const double lowerQuartile =
          calc_median(sortedTimings.begin(), sortedTimings.begin() + sortedTimings.size() / 2);
      out << std::right << std::setw(format.space) << format_time(lowerQuartile);
    } break;
    case Stat::Min:
      out << std::right << std::setw(format.space) << format_time(sortedTimings.front());
      break;
    case Stat::Max:
      out << std::right << std::setw(format.space) << format_time(sortedTimings.back());
      break;
    case Stat::Percentage: {
      const double p =
          (totalSum < currentSum || totalSum == 0) ? 100.0 : currentSum / totalSum * 100.0;
      out << std::right << std::fixed << std::setprecision(2) << std::setw(format.space) << p;
    } break;
    case Stat::ParentPercentage: {
      const double p =
          (parentSum < currentSum || parentSum == 0) ? 100.0 : currentSum / parentSum * 100.0;
      out << std::right << std::fixed << std::setprecision(2) << std::setw(format.space) << p;
    } break;
  }
}

// Helper struct for creating a tree of timings
struct TimeStampPair {
  std::string identifier;
  double time = 0.0;
  std::size_t startIdx = 0;
  std::size_t stopIdx = 0;
  internal::TimingNode* nodePtr = nullptr;
};

auto calculate_statistic(std::vector<double> values)
    -> std::tuple<double, double, double, double, double, double, double> {
  if (values.empty()) return std::make_tuple(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  std::sort(values.begin(), values.end());

  const double min = values.front();
  const double max = values.back();

  const double median = calc_median(values.begin(), values.end());
  const double sum = std::accumulate(values.begin(), values.end(), 0.0);
  const double mean = sum / values.size();

  const double lowerQuartile = calc_median(values.begin(), values.begin() + values.size() / 2);
  const double upperQuartile = calc_median(
      values.begin() + values.size() / 2 + (values.size() % 2) * (values.size() > 1), values.end());

  return std::make_tuple(sum, mean, median, min, max, lowerQuartile, upperQuartile);
}

// print rt_graph nodes in tree recursively
auto print_node(std::ostream& out, const std::vector<internal::Format> formats,
                const std::size_t identifierSpace, const std::string& nodePrefix,
                const internal::TimingNode& node, const bool isSubNode, const bool isLastSubnode,
                double parentTime, double totalTime) -> void {
  double sum, mean, median, min, max, lowerQuartile, upperQuartile;
  std::tie(sum, mean, median, min, max, lowerQuartile, upperQuartile) =
      calculate_statistic(node.timings);

  if (!isSubNode) {
    totalTime = sum;
    parentTime = sum;
  }

  const double totalPercentage =
      (totalTime < sum || totalTime == 0) ? 100.0 : sum / totalTime * 100.0;

  const double parentPercentage =
      (parentTime < sum || parentTime == 0) ? 100.0 : sum / parentTime * 100.0;

  std::stringstream totalPercentageStream;
  totalPercentageStream << std::fixed << std::setprecision(2) << totalPercentage;
  std::stringstream parentPercentageStream;
  parentPercentageStream << std::fixed << std::setprecision(2) << parentPercentage;

  out << std::left << std::setw(identifierSpace);
  if (isSubNode)
    out << nodePrefix + "- " + node.identifier;
  else
    out << nodePrefix + node.identifier;

  auto sortedTimings = node.timings;
  std::sort(sortedTimings.begin(), sortedTimings.end());

  const double currentTime = std::accumulate(sortedTimings.begin(), sortedTimings.end(), 0.0);
  for (const auto& format : formats) {
    print_stat(out, format, sortedTimings, totalTime, parentTime, currentTime);
  }

  out << std::endl;

  for (const auto& subNode : node.subNodes) {
    print_node(out, formats, identifierSpace, nodePrefix + std::string(" |"), subNode, true,
               &subNode == &node.subNodes.back(), sum, totalTime);
    if (!isLastSubnode && &subNode == &node.subNodes.back()) {
      out << nodePrefix << std::endl;
    }
  }
}

// determine length of padding required for printing entire tree identifiers recursively
auto max_node_identifier_length(const internal::TimingNode& node, const std::size_t recursionDepth,
                                const std::size_t addPerLevel, const std::size_t parentMax)
    -> std::size_t {
  std::size_t currentLength = node.identifier.length() + recursionDepth * addPerLevel;
  std::size_t max = currentLength > parentMax ? currentLength : parentMax;
  for (const auto& subNode : node.subNodes) {
    const std::size_t subMax =
        max_node_identifier_length(subNode, recursionDepth + 1, addPerLevel, max);
    if (subMax > max) max = subMax;
  }

  return max;
}

auto export_node_json(const std::string& padding, const std::list<internal::TimingNode>& nodeList,
                      std::ostream& stream) -> void {
  stream << "{" << std::endl;
  const std::string nodePadding = padding + "  ";
  const std::string subNodePadding = nodePadding + "  ";
  for (const auto& node : nodeList) {
    stream << nodePadding << "\"" << node.identifier << "\" : {" << std::endl;
    stream << subNodePadding << "\"timings\" : [";
    for (const auto& value : node.timings) {
      stream << value;
      if (&value != &(node.timings.back())) stream << ", ";
    }
    stream << "]," << std::endl;
    stream << subNodePadding << "\"sub-timings\" : ";
    export_node_json(subNodePadding, node.subNodes, stream);
    stream << nodePadding << "}";
    if (&node != &(nodeList.back())) stream << ",";
    stream << std::endl;
  }
  stream << padding << "}" << std::endl;
}

auto extract_timings(const std::string& identifier, const std::list<TimingNode>& nodes,
                     std::vector<double>& timings) -> void {
  for (const auto& node : nodes) {
    if (node.identifier == identifier) {
      timings.insert(timings.end(), node.timings.begin(), node.timings.end());
    }
    extract_timings(identifier, node.subNodes, timings);
  }
}

}  // namespace
}  // namespace internal

// ======================
// Timer
// ======================
auto Timer::process() const -> TimingResult {
  std::list<internal::TimingNode> results;
  std::stringstream warnings;

  try {
    std::vector<internal::TimeStampPair> timePairs;
    timePairs.reserve(timeStamps_.size() / 2);

    // create pairs of start / stop timings
    for (std::size_t i = 0; i < timeStamps_.size(); ++i) {
      if (timeStamps_[i].type == internal::TimeStampType::Start) {
        internal::TimeStampPair pair;
        pair.startIdx = i;
        pair.identifier = std::string(timeStamps_[i].identifierPtr);
        std::size_t numInnerMatchingIdentifiers = 0;
        // search for matching stop after start
        for (std::size_t j = i + 1; j < timeStamps_.size(); ++j) {
          // only consider matching identifiers
          if (std::string(timeStamps_[j].identifierPtr) ==
              std::string(timeStamps_[i].identifierPtr)) {
            if (timeStamps_[j].type == internal::TimeStampType::Stop &&
                numInnerMatchingIdentifiers == 0) {
              // Matching stop found
              std::chrono::duration<double> duration = timeStamps_[j].time - timeStamps_[i].time;
              pair.time = duration.count();
              pair.stopIdx = j;
              timePairs.push_back(pair);
              if (pair.time < 0) {
                warnings << "rt_graph WARNING:Measured time is negative. Non-steady system-clock?!"
                         << std::endl;
              }
              break;
            } else if (timeStamps_[j].type == internal::TimeStampType::Stop &&
                       numInnerMatchingIdentifiers > 0) {
              // inner stop with matching identifier
              --numInnerMatchingIdentifiers;
            } else if (timeStamps_[j].type == internal::TimeStampType::Start) {
              // inner start with matching identifier
              ++numInnerMatchingIdentifiers;
            }
          }
        }
        if (pair.stopIdx == 0) {
          warnings << "rt_graph WARNING: Start / stop time stamps do not match for \""
                   << timeStamps_[i].identifierPtr << "\"!" << std::endl;
        }
      }
    }

    // create tree of timings where sub-nodes represent timings fully enclosed by another start /
    // stop pair Use the fact that timePairs is sorted by startIdx
    for (std::size_t i = 0; i < timePairs.size(); ++i) {
      auto& pair = timePairs[i];

      // find potential parent by going backwards through pairs, starting with the current pair
      // position
      for (auto timePairIt = timePairs.rbegin() + (timePairs.size() - i);
           timePairIt != timePairs.rend(); ++timePairIt) {
        if (timePairIt->stopIdx > pair.stopIdx && timePairIt->nodePtr != nullptr) {
          auto& parentNode = *(timePairIt->nodePtr);
          // check if sub-node with identifier exists
          bool nodeFound = false;
          for (auto& subNode : parentNode.subNodes) {
            if (subNode.identifier == pair.identifier) {
              nodeFound = true;
              subNode.timings.push_back(pair.time);
              // mark node position in pair for finding sub-nodes
              pair.nodePtr = &(subNode);
              break;
            }
          }
          if (!nodeFound) {
            // create new sub-node
            internal::TimingNode newNode;
            newNode.identifier = pair.identifier;
            newNode.timings.push_back(pair.time);
            parentNode.subNodes.push_back(std::move(newNode));
            // mark node position in pair for finding sub-nodes
            pair.nodePtr = &(parentNode.subNodes.back());
          }
          break;
        }
      }

      // No parent found, must be top level node
      if (pair.nodePtr == nullptr) {
        // Check if top level node with same name exists
        for (auto& topNode : results) {
          if (topNode.identifier == pair.identifier) {
            topNode.timings.push_back(pair.time);
            pair.nodePtr = &(topNode);
            break;
          }
        }
      }

      // New top level node
      if (pair.nodePtr == nullptr) {
        internal::TimingNode newNode;
        newNode.identifier = pair.identifier;
        newNode.timings.push_back(pair.time);
        // newNode.parent = nullptr;
        results.push_back(std::move(newNode));

        // mark node position in pair for finding sub-nodes
        pair.nodePtr = &(results.back());
      }
    }
  } catch (const std::exception& e) {
    warnings << "rt_graph WARNING: Processing of timings failed: " << e.what() << std::endl;
  } catch (...) {
    warnings << "rt_graph WARNING: Processing of timings failed!" << std::endl;
  }

  return TimingResult(std::move(results), warnings.str());
}

// ======================
//
// ======================

auto TimingResult::json() const -> std::string {
  std::stringstream jsonStream;
  jsonStream << std::scientific;
  internal::export_node_json("", rootNodes_, jsonStream);
  return jsonStream.str();
}

auto TimingResult::get_timings(const std::string& identifier) const -> std::vector<double> {
  std::vector<double> timings;
  internal::extract_timings(identifier, rootNodes_, timings);
  return timings;
}

auto TimingResult::print(std::vector<Stat> statistic) const -> std::string {
  std::stringstream stream;

  // print warnings
  stream << warnings_;

  // calculate space for printing identifiers
  std::size_t identifierSpace = 0;
  for (const auto& node : rootNodes_) {
    const auto nodeMax = internal::max_node_identifier_length(node, 0, 2, identifierSpace);
    if (nodeMax > identifierSpace) identifierSpace = nodeMax;
  }
  identifierSpace += 3;

  auto totalSpace = identifierSpace;

  std::vector<internal::Format> formats;
  formats.reserve(statistic.size());
  for (const auto& stat : statistic) {
    formats.emplace_back(stat);
    totalSpace += formats.back().space;
  }

  // Construct table header

  // Table start
  stream << std::string(totalSpace, '=') << std::endl;

  // header
  stream << std::right << std::setw(identifierSpace) << "";
  for (const auto& format : formats) {
    stream << std::right << std::setw(format.space) << format.header;
  }
  stream << std::endl;

  // Header separation line
  stream << std::string(totalSpace, '-') << std::endl;

  // print all timings
  for (const auto& node : rootNodes_) {
    internal::print_node(stream, formats, identifierSpace, std::string(), node, false, true, 0.0,
                         0.0);
    stream << std::endl;
  }

  // End table
  stream << std::string(totalSpace, '=') << std::endl;

  return stream.str();
}

}  // namespace rt_graph

