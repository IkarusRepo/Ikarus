/*
 * This file is part of the Ikarus distribution (https://github.com/IkarusRepo/Ikarus).
 * Copyright (c) 2022. The Ikarus developers.
 *
 * This library is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU Lesser General Public
 *  License as published by the Free Software Foundation; either
 *  version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA
 */

#include "feRequirements.hh"

namespace Ikarus {

  std::string getResultType(const ResultType& res) {
    switch (res) {
      case ResultType::noType:
        return "noType";
      case ResultType::magnetization:
        return "magnetization";
      case ResultType::vectorPotential:
        return "vectorPotential";
      case ResultType::gradientNormOfMagnetization:
        return "gradientNormOfMagnetization";
        break;
      case ResultType::divergenceOfVectorPotential:
        return "divergenceOfVectorPotential";
        break;
      case ResultType::cauchyStress:
        return "cauchyStress";
        break;
      case ResultType::director:
        return "director";
        break;
      case ResultType::BField:
        return "BField";
        break;
      case ResultType::HField:
        return "HField";
        break;
      case ResultType::stressResultant:
        return "stressResultant";
        break;
    }
    __builtin_unreachable();
  }

}  // namespace Ikarus
