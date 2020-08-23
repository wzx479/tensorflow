// predictTF.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#pragma once

#define COMPILER_MSVC
#define NOMINMAX

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

//#include "TensorflowObjectDetection.h"
#include <utility>
#include <fstream>
#include <regex>
#include <iostream>
#include <utility>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

// 最大参数
int ArgMax(const tensorflow::TTypes<float, 1>::Tensor& prediction)
{
	float max_value = -1.0;
	int max_index = -1;
	const long count = prediction.size();
	for (int i = 0; i < count; ++i) {
		const float value = prediction(i);
		if (value > max_value) {
			max_index = i;
			max_value = value;
		}
		std::cout << "value[" << i << "] = " << value << std::endl;
	}
	return max_index;
}

std::string class_names[] = { "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot" };

// Mat转TensorFlow
tensorflow::Tensor Mat2Tensor(cv::Mat &img, float normal = 1 / 255.0) {
	tensorflow::Tensor image_input = tensorflow::Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(
		{ 1, img.size().height, img.size().width, img.channels() }));
	float *tensor_data_ptr = image_input.flat<float>().data();
	cv::Mat fake_mat(img.rows, img.cols, CV_32FC(img.channels()), tensor_data_ptr);
	img.convertTo(fake_mat, CV_32FC(img.channels()));
	fake_mat *= normal;
	return image_input;
}


int main(int argc, char* argv[]) {

	// 初始化 tensorflow Session
	tensorflow::Session* session;
	tensorflow::Status status = NewSession(tensorflow::SessionOptions(), &session);
	if (!status.ok()) {
		std::cerr << status.ToString() << "session status no ok" << std::endl;
		std::cin.get();
		return -1;
	}
	else {
		std::cout << "Session created successfully" << std::endl;
	}

	// Load the protobuf graph   // 暂时理解为权重文件
	tensorflow::GraphDef graph_def;

	std::string graph_path = "D:/Desktop/frozen_model.pb"; // 暂时理解为权重文件   mmmmmm.h5.pb   tf1.2.pb   !!! 自己的模型 .pb 用tf1.x转
	status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), graph_path, &graph_def);
	if (!status.ok()) {
		std::cerr << status.ToString() << "********************" << std::endl;
		system("pause");
		return 1;
	}
	else {
		std::cout << "Load graph protobuf successfully" << std::endl;
	}


	// 读取图片
	cv::Mat image = cv::imread("D:/desktop/6.jpg", cv::IMREAD_GRAYSCALE);   // 要预测的图片
	tensorflow::Tensor input_image = Mat2Tensor(image, 1 / 255.0);

	// 将 graph 转为 session
	status = session->Create(graph_def);
	if (!status.ok()) {
		std::cerr << status.ToString() << std::endl;
		system("pause");
		return 1;
	}
	else {
		std::cout << "Add graph to session successfully!" << std::endl;
	}


	// Setup inputs and outputs:  设置输入输出
	std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
		{ "prefetch_queue/fifo_queue:0", input_image }  //  input  input_image_input  // .pb文件用Netron 打开后的输入层
	};

	// The session will initialize the outputs  // session 初始化输出
	std::vector<tensorflow::Tensor> outputs;

	// Run the session, evaluating our "c" operation from the graph
	status = session->Run(inputs, { "InceptionV3/Logits/SpatialSqueeze:0" }, {}, &outputs);  //.pb文件用Netron 打开后的输出层

	if (!status.ok()) {
		std::cerr << status.ToString() << std::endl;
		std::cout << "session run 失败" << std::endl;
		system("pause");
		return 1;
	}
	else {
		std::cout << "Run session successfully" << std::endl;
	}

	// Grab the first output (we only evaluated one graph node: "c")
	// and convert the node to a scalar representation.
	// Print the results
	std::cout << outputs[0].DebugString() << std::endl; // Tensor<type: float shape: [] values: 30>

	// const Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>, Eigen::Aligned>& prediction = outputs[0].flat<float>();
	const tensorflow::TTypes<float, 1>::Tensor& prediction = outputs[0].flat_inner_dims<float, 1>();


	// 预测的标签
	int pred_index = ArgMax(prediction);

	// Print test accuracy 看一下精准度
	printf("Predict: %d Label: %s", pred_index, class_names[pred_index].c_str());


	// Free any resources used by the session // 释放资源
	session->Close();
	system("pause");
	return 0;
}