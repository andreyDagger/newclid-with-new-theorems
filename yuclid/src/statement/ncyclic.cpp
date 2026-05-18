/**
   Copyright 2025 Concordance Inc. dba Harmonic

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#include "ncyclic.hpp"
#include "coll.hpp"
#include "statement/statement.hpp"
#include "type/point.hpp"
#include "typedef.hpp"
#include <algorithm>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

namespace Yuclid {

  NonCyclicQuadrangle::NonCyclicQuadrangle(Point a, Point b, Point c, Point d)
    : m_a(a), m_b(b), m_c(c), m_d(d) {}

  NonCyclicQuadrangle::NonCyclicQuadrangle(const vector<statement_arg>& args) :
    m_a(get<Point>(args.at(0))), m_b(get<Point>(args.at(1))),
    m_c(get<Point>(args.at(2))), m_d(get<Point>(args.at(3)))
  {
    if (args.size() != 4) {
      throw invalid_argument("ncyclic constructor expects 4 arguments.");
    }
  }

  string NonCyclicQuadrangle::name() const { return "ncyclic"; }

  vector<Point> NonCyclicQuadrangle::points() const {
    return {m_a, m_b, m_c, m_d};
  }

  unique_ptr<Statement> NonCyclicQuadrangle::normalize() const {
    vector<Point> pts = {m_a, m_b, m_c, m_d};
    ranges::sort(pts);
    return make_unique<NonCyclicQuadrangle>(pts[0], pts[1], pts[2], pts[3]);
  }

  bool NonCyclicQuadrangle::check_nondegen() const {
    return equal_angles_cad_cbd().check_nondegen() &&
      equal_angles_bad_bcd().check_nondegen() &&
      !Collinear(m_a, m_b, m_c).check_equations();
  }

  bool NonCyclicQuadrangle::check_equations() const {
    return !equal_angles_cad_cbd().check_equations();
  }

  vector<statement_arg> NonCyclicQuadrangle::args() const {
    return {m_a, m_b, m_c, m_d};
  }

  unique_ptr<Statement> NonCyclicQuadrangle::clone() const {
    return make_unique<NonCyclicQuadrangle>(*this);
  }

  // Implementation of statement-specific methods for cyclic
  EqualAngles NonCyclicQuadrangle::equal_angles_cad_cbd() const {
    return {Angle(m_c, m_a, m_d), Angle(m_c, m_b, m_d)};
  }

  EqualAngles NonCyclicQuadrangle::equal_angles_bad_bcd() const {
    return {Angle(m_b, m_a, m_d), Angle(m_b, m_c, m_d)};
  }

  EqualAngles NonCyclicQuadrangle::equal_angles_abd_acd() const {
    return {Angle(m_a, m_b, m_d), Angle(m_a, m_c, m_d)};
  }

  ostream &NonCyclicQuadrangle::print(ostream &out) const {
    return out << m_a << " not ∈ ω(" << m_b << m_c << m_d << ")";
  }

} // namespace Yuclid
