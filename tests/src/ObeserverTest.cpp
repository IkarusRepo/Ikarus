//
// Created by Alex on 21.07.2021.
//
//#include <gmock/gmock.h>
//#include <gtest/gtest.h>

//#include "testHelpers.h"

#include "ikarus/utils/Observer/controlLogger.h"
#include <catch2/catch_test_macros.hpp>
class Control : public IObservable<ControlMessages> {
public:
  void solve() {}
};

TEST_CASE("Observer: ControlObserver", "[1]") {
  auto controlObserver = std::make_shared<ControlLogger>();
  Control control;
  control.subscribeAll(controlObserver);

  control.notify(ControlMessages::CONTROL_STARTED);
  control.notify(ControlMessages::STEP_ENDED);
  control.notify(ControlMessages::SOLUTION_CHANGED);
}
