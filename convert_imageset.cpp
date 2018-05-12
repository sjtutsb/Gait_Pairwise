// This program converts a set of images to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"


#include "opencv2/core/core.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

DEFINE_bool(gray, false,
	"When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
	"Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "lmdb",
	"The backend {lmdb, leveldb} for storing the result");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_bool(check_size, false,
	"When this option is on, check that all the datum have the same size");
DEFINE_bool(encoded, false,
	"When this option is on, the encoded image will be save in datum");
DEFINE_string(encode_type, "",
	"Optional: What type should we encode the image as ('png','jpg',...).");

typedef struct image_pair{
	std::string image1;
	std::string image2;
	int label;
}image_pair;




int main(int argc, char** argv) {
#ifdef USE_OPENCV
	::google::InitGoogleLogging(argv[0]);
	// Print output to stderr (while still logging)
	FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
	namespace gflags = google;
#endif

	gflags::SetUsageMessage("Convert a set of images to the leveldb/lmdb\n"
		"format used as input for Caffe.\n"
		"Usage:\n"
		"    convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n"
		"The ImageNet dataset for the training demo is at\n"
		"    http://www.image-net.org/download-images\n");
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	if (argc < 4) {
		gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_imageset");
		return 1;
	}

	const bool is_color = !FLAGS_gray;
	const bool check_size = FLAGS_check_size;
	const bool encoded = FLAGS_encoded;
	const string encode_type = FLAGS_encode_type;

	std::ifstream infile(argv[2]);
	std::vector<image_pair> lines;
	std::string line;
	while (std::getline(infile, line)) {
		size_t pos;
		image_pair tmp;
		pos = line.find(' ');
		tmp.image1 = line.substr(0, pos);
		line = line.substr(pos + 1);
		pos = line.find(' ');
		tmp.image2 = line.substr(0, pos);
		tmp.label = atoi(line.substr(pos + 1).c_str());
		lines.push_back(tmp);
	}
	if (FLAGS_shuffle) {
		// randomly shuffle data
		LOG(INFO) << "Shuffling data";
		shuffle(lines.begin(), lines.end());
	}
	LOG(INFO) << "A total of " << lines.size() << " images.";

	if (encode_type.size() && !encoded)
		LOG(INFO) << "encode_type specified, assuming encoded=true.";

	int resize_height = std::max<int>(0, FLAGS_resize_height);
	int resize_width = std::max<int>(0, FLAGS_resize_width);

	// Create new DB
	scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
	db->Open(argv[3], db::NEW);
	scoped_ptr<db::Transaction> txn(db->NewTransaction());

	// Storing to db
	std::string root_folder(argv[1]);
	Datum datum;
	int count = 0;
	int data_size = 0;
	bool data_size_initialized = false;

	for (int line_id = 0; line_id < lines.size(); ++line_id) {
		std::string enc = encode_type;
		if (encoded && !enc.size()) {
			// Guess the encoding type from the file name
			string fn = lines[line_id].image1;
			size_t p = fn.rfind('.');
			if (p == fn.npos)
				LOG(WARNING) << "Failed to guess the encoding of '" << fn << "'";
			enc = fn.substr(p);
			std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
		}//这里没对所有图片进行处理，不会运行这里。


		cv::Mat cv_img1 = ReadImageToCVMat(root_folder + lines[line_id].image1, resize_height, resize_width, is_color);
		cv::Mat cv_img2 = ReadImageToCVMat(root_folder + lines[line_id].image2, resize_height, resize_width, is_color);
		if (cv_img1.data&&cv_img2.data)
		{
			datum.set_channels(cv_img1.channels() + cv_img2.channels());
			datum.set_height(resize_height);
			datum.set_width(resize_width);
			datum.clear_data();
			datum.clear_float_data();
			datum.set_encoded(false);
			int datum_channels = datum.channels();
			int datum_height = datum.height();
			int datum_width = datum.width();
			int datum_size = datum_channels * datum_height * datum_width;
			std::string buffer(datum_size, ' ');
			for (int h = 0; h < datum_height; ++h) {
				const uchar* ptr = cv_img1.ptr<uchar>(h);
				int img_index = 0;
				for (int w = 0; w < datum_width; ++w) {
					for (int c = 0; c < cv_img1.channels(); ++c) {
						int datum_index = (c * datum_height + h) * datum_width + w;
						buffer[datum_index] = static_cast<char>(ptr[img_index++]);
					}
				}
			}
			for (int h = 0; h < datum_height; ++h) {
				const uchar* ptr = cv_img2.ptr<uchar>(h);
				int img_index = 0;
				for (int w = 0; w < datum_width; ++w) {
					for (int c = cv_img1.channels(); c < datum.channels(); ++c) {
						int datum_index = (c * datum_height + h) * datum_width + w;
						buffer[datum_index] = static_cast<char>(ptr[img_index++]);
					}
				}
			}
			datum.set_data(buffer);
			datum.set_label(lines[line_id].label);
		}
		else
		{
			continue;
		}
		if (check_size) {
			if (!data_size_initialized) {
				data_size = datum.channels() * datum.height() * datum.width();
				data_size_initialized = true;
			}
			else {
				const std::string& data = datum.data();
				CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
					<< data.size();
			}
		}
		// sequential
		string key_str = caffe::format_int(line_id, 8) + "_" + lines[line_id].image1;

		// Put in db
		string out;
		CHECK(datum.SerializeToString(&out));
		txn->Put(key_str, out);

		if (++count % 1000 == 0) {
			// Commit db
			txn->Commit();
			txn.reset(db->NewTransaction());
			LOG(INFO) << "Processed " << count << " files.";
		}
	}
	// write the last batch
	if (count % 1000 != 0) {
		txn->Commit();
		LOG(INFO) << "Processed " << count << " files.";
	}
#else
	LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
	return 0;
}