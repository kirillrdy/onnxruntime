// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/coreml/builders/helper.h"
#include "core/providers/coreml/builders/impl/base_op_builder.h"
#include "core/providers/coreml/builders/impl/builder_utils.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"
#include "core/providers/coreml/shape_utils.h"
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime {
namespace coreml {

class TransposeOpBuilder : public BaseOpBuilder {
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;

  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                         const logging::Logger& logger) const override;

  bool SupportsMLProgram() const override { return true; }

  bool IsTrivial(const Node& /*node*/) const override { return true; }
};

Status TransposeOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                 const Node& node,
                                                 const logging::Logger& logger) const {
  NodeAttrHelper helper(node);
  std::vector<int64_t> perm = helper.Get("perm", std::vector<int64_t>());
  std::vector<int64_t> input_shape;
  ORT_RETURN_IF_NOT(GetShape(*node.InputDefs()[0], input_shape, logger), "Cannot get shape");
  auto input_dims = input_shape.size();
  if (perm.empty()) {
    for (int64_t i = input_dims - 1; i >= 0; i--)
      perm.push_back(i);
  } else {
    ORT_RETURN_IF_NOT(perm.size() == input_dims, "Perm and input should have same dimension");
  }

  if (model_builder.CreateMLProgram()) {
    using namespace CoreML::Specification::MILSpec;

    if (perm.size() == 6 && perm[0] == 0 && perm[1] == 1) {
      // Lower 6D perm (0, 1, p2, p3, p4, p5) to 5D perm (0, p2-1, p3-1, p4-1, p5-1)
      // to stay within CoreML rank <= 5 limits.
      std::vector<int64_t> perm_5d = {
          0,
          perm[2] - 1,
          perm[3] - 1,
          perm[4] - 1,
          perm[5] - 1,
      };
      std::unique_ptr<Operation> op = model_builder.CreateOperation(node, "transpose");
      AddOperationInput(*op, "x", node.InputDefs()[0]->Name());
      AddOperationInput(*op, "perm", model_builder.AddConstant(op->type(), "perm", perm_5d));

      std::vector<int64_t> input_shape_5d = {
          input_shape[0] * input_shape[1],
          input_shape[2],
          input_shape[3],
          input_shape[4],
          input_shape[5],
      };
      std::vector<int64_t> output_shape_5d(5);
      for (size_t i = 0; i < 5; ++i) {
        output_shape_5d[i] = input_shape_5d[perm_5d[i]];
      }
      const auto dtype = node.OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
      AddIntermediateOperationOutput(*op, node.OutputDefs()[0]->Name(), dtype, output_shape_5d);

      model_builder.AddOperation(std::move(op));
      return Status::OK();
    }

    std::unique_ptr<Operation> op = model_builder.CreateOperation(node, "transpose");
    AddOperationInput(*op, "x", node.InputDefs()[0]->Name());
    AddOperationInput(*op, "perm", model_builder.AddConstant(op->type(), "perm", perm));
    AddOperationOutput(*op, *node.OutputDefs()[0]);
    model_builder.AddOperation(std::move(op));

  } else {
    std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> layer = model_builder.CreateNNLayer(node);
    *layer->mutable_transpose()->mutable_axes() = {perm.cbegin(), perm.cend()};

    *layer->mutable_input()->Add() = node.InputDefs()[0]->Name();
    *layer->mutable_output()->Add() = node.OutputDefs()[0]->Name();

    model_builder.AddLayer(std::move(layer));
  }
  return Status::OK();
}

bool TransposeOpBuilder::IsOpSupportedImpl(const Node& node,
                                           const OpBuilderInputParams& input_params,
                                           const logging::Logger& logger) const {
  std::vector<int64_t> input_shape;
  if (!GetShape(*node.InputDefs()[0], input_shape, logger)) {
    return false;
  }

  if (input_shape.size() > 5) {
    if (!input_params.create_mlprogram) {
      LOGS(logger, VERBOSE) << "Transpose does not support input rank greater than 5 in NeuralNetwork format";
      return false;
    }
    if (input_shape.size() == 6) {
      NodeAttrHelper helper(node);
      std::vector<int64_t> perm = helper.Get("perm", std::vector<int64_t>());
      if (perm.size() != 6 || perm[0] != 0 || perm[1] != 1) {
        LOGS(logger, VERBOSE) << "6D Transpose only supported when batch and leading dimension are preserved (perm[0]==0, perm[1]==1)";
        return false;
      }
    } else {
      LOGS(logger, VERBOSE) << "Transpose does not support input rank greater than 6";
      return false;
    }
  }
  return true;
}

void CreateTransposeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<TransposeOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace coreml
}  // namespace onnxruntime
