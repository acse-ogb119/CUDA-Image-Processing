#pragma once

void convert_grayscale(const std::string &input_file, const std::string &output_file);
void gaussian_blur(const std::string &input_file, const std::string &output_file);
void tonemap_HDR(const std::string &input_file, const std::string &output_file);
void remove_redeye(const std::string &input_file, const std::string &template_file, const std::string &output_file);
void seamless_clone(const std::string &input_file, const std::string &dest_file, const std::string &output_file);
