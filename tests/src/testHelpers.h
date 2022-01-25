//
// Created by Alex on 20.05.2021.
//

#pragma once

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>

#include <string>
#include <iostream>
#include <sstream>

template<typename Derived>
struct EigenApproxEqual : Catch::Matchers::MatcherGenericBase {
  EigenApproxEqual(const Derived& xDerived, double tol):
      xDerived{ xDerived }, tol{tol}
  {}

  template<typename Derived2>
  bool match(const Derived2& other) const {

    return xDerived.isApprox(other, tol);
  }

  std::string describe() const override {
    std::stringstream buffer;
    buffer << xDerived << std::endl;
    return "Equals: \n" + buffer.str();
  }

 private:
  Derived const& xDerived;
  double tol;
};


template<typename Derived>
struct EigenExactEqual : Catch::Matchers::MatcherGenericBase {
  EigenExactEqual(const Derived& xDerived):
      xDerived{ xDerived }
  {}

  template<typename Derived2>
  bool match(const Derived2& other) const {

    return ((xDerived == other) == true).all();
  }

  std::string describe() const override {
    std::stringstream buffer;
    buffer << xDerived << std::endl;
    return "Equals: \n" + buffer.str();
  }

 private:
  Derived const& xDerived;
};



//MATCHER_P2(EigenApproxEqual, expect, prec,
//           std::string(negation ? "isn't" : "is") + " approx equal to" + ::testing::PrintToString(expect)
//               + "\nwith precision " + ::testing::PrintToString(prec)) {
//  return arg.isApprox(expect, prec);
//}
//
//MATCHER_P(EigenExactEqual, expect,
//          std::string(negation ? "isn't" : "is") + " equal to" + ::testing::PrintToString(expect)) {
//  return ((arg == expect) == true).all();
//}
