/**
 * Conversely, if you can't train a classifier to tell the difference between real
 * and generated data even for the initial random generator output, you can't get
 * the GAN training started.
 *
 */

#include"custom_modules.hpp"
#include"custom_datasets.hpp"
#include"utils/yaml-torch.hpp"
#include"utils/png-torch.hpp"

using namespace custom_models;
using namespace custom_models::datasets;
using namespace torch::data::datasets;
using namespace torch::indexing;

	template <typename generator>
void test(generator& gen, const int64_t num_of_pictures,const int64_t width,const int64_t height,torch::Device device,const size_t &epoch)
{
	static auto z = torch::randn({num_of_pictures, gen->input_size},device);
	auto G_result = gen(z);

	png_interface::write_png_file(("PICTURES/Test_"+std::to_string(epoch)+".png").c_str(),
			G_result.view({num_of_pictures,width,height}));
}
template <typename DataLoader, typename generator,typename discriminator>
void train(size_t epoch, generator& gen, discriminator &dis,torch::Device device, DataLoader& data_loader,YAML::Node config) {


	static std::ofstream RFile("Results.txt", std::ios::out | std::ios::app);
	auto bcel=torch::nn::BCELoss();
	dis->train();

	auto Doptimizer=yaml_interface::get_optimizer(config["Optimizer"],dis->parameters());
	auto Goptimizer=yaml_interface::get_optimizer(config["Optimizer"],gen->parameters());
	const static auto num_batches=(config["Train"])["Number of batches"].as<int64_t>();


	size_t iterator=0;
	size_t dataset_size=0;

	double sumLossD=0.0,sumLossG=0.0;
	int64_t correctD = 0,correctG=0;
	for (auto& batch : data_loader) {

		if(iterator>=num_batches)break;
		//train discriminator
		auto x = batch.data.to(device).to(at::get_default_dtype_as_scalartype());
		const auto batch_size=x.size(0);
		dataset_size+=batch_size;
		dis->zero_grad();


		auto D_result = dis(x);

		if(D_result.size(1)>1)D_result=torch::abs(D_result.index({Slice(),Slice(0,1)}));

		const auto y_real = torch::ones_like(D_result,device);
		const auto y_fake = torch::zeros_like(D_result,device);

		auto bin_D_result=(D_result>0.5).to(torch::kInt64);

		correctD += bin_D_result.sum().template item<int64_t>();

		auto D_real_loss = bcel(D_result, y_real);


		auto z = torch::randn({batch_size, gen->input_size},device);

		auto G_result = gen(z);

		auto D_resultf = dis(G_result);

		if(D_resultf.size(1)>1)D_resultf=torch::abs(D_resultf.index({Slice(),Slice(None,1)}));

		auto bin_D_resultf=(D_resultf<=0.5).to(torch::kInt64);

		correctD += bin_D_resultf.sum().template item<int64_t>();


		auto D_fake_loss = bcel(D_resultf, y_fake);

		auto D_train_loss = D_real_loss + D_fake_loss;

		D_train_loss.backward();
		Doptimizer->step();

		sumLossD+=D_train_loss.template item<double>();
		// train generator G

		gen->zero_grad();

		auto z2 = torch::randn({batch_size, gen->input_size},device);

		auto G_result2 = gen(z2);

		auto D_result2 = dis(G_result2);
		if(D_result2.size(1)>1)D_result2=torch::abs(D_result2.index({Slice(),Slice(None,1)}));


		auto y = torch::ones_like(D_result2,device);
		auto bin_D_result2=(D_result2>0.5).to(torch::kInt64);

		correctG += bin_D_result2.sum().template item<int64_t>();

		auto G_train_loss = bcel(D_result2, y);
		G_train_loss.backward();
		Goptimizer->step();

		sumLossG+=G_train_loss.template item<double>();

		iterator++;
	}
	dis->update();
	const auto AccuG=1.0*correctG/dataset_size;
	const auto AccuD=correctD/2.0/dataset_size;

	RFile<<std::setw(12)<<sumLossG/dataset_size<<std::setw(12)<<sumLossD/dataset_size
		<<std::setw(12)<<AccuG<<std::setw(12)<<AccuD<<std::endl;

}



int main(int argc, char** argv)
{
	torch::manual_seed(0);
	YAML::Node config = YAML::LoadFile(((argc>1)?argv[1]:"config.yaml"));
	const auto USE_GPU=config["USE_GPU"].as<bool>();
	torch::DeviceType device_type= torch::kCPU;
	if(USE_GPU)
	{
		if (torch::cuda::is_available()) {
			std::cout << "Device: GPU." << std::endl;
			device_type = torch::kCUDA;
		} else {
			std::cout << "Device: CPU." << std::endl;
		}
	}
	else
	{
		std::cout << "Device: CPU." << std::endl;
	}
	torch::Device device(device_type);

	at::set_default_dtype(caffe2::TypeMeta::Make<double>());


	auto generator=GENERATOR(config["Generator"]);

	auto discriminator=DISCRIMINATOR(config["Discriminator"]);

	if((config["Load and Save Module"])["Restart"].as<bool>())
	{
		std::cout<<"Loading module from file "<<(config["Load and Save Module"])["From"].as<std::string>()<<std::endl;
		torch::load(generator,(config["Load and Save Module"])["From"].as<std::string>()+"generator.pt");
		torch::load(discriminator,(config["Load and Save Module"])["From"].as<std::string>()+"discriminator.pt");
	}
	generator->to(device);
	discriminator->to(device);

	auto train_dataset = DATASET((config["Dataset"])["From"].as<std::string>())
		.map(torch::data::transforms::Normalize<>(0.5, 0.5))
		.map(torch::data::transforms::Stack<>());

	auto train_loader =
		torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
				std::move(train_dataset),(config["Train"])["Batch size"].as<size_t>() );


	for (size_t epoch = 1; epoch <= config["Number of epochs"].as<size_t>() ; ++epoch)
	{
		std::cout<<"Epoch:"<<epoch<<std::endl;
#if TRAIN
		train(epoch,generator,discriminator,device,*train_loader,config);
		if(epoch%((config["Load and Save Module"])["Save every"].as<size_t>())==0)
		{
			std::cout<<"Saving model to "<<(config["Load and Save Module"])["To"].as<std::string>()<<std::endl;
			torch::save(discriminator, (config["Load and Save Module"])["To"].as<std::string>()+"discriminator.pt");
			torch::save(generator, (config["Load and Save Module"])["To"].as<std::string>()+"generator.pt");
		}
#endif
#if TEST
		test(generator,(config["Test"])["Batch size"].as<int64_t>(),
				(config["Test"])["image width"].as<int64_t>(),
				(config["Test"])["image height"].as<int64_t>(),device,epoch);
#endif

	}
#if TRAIN	
	torch::save(discriminator, (config["Load and Save Module"])["To"].as<std::string>()+"discriminator.pt");
	torch::save(generator, (config["Load and Save Module"])["To"].as<std::string>()+"generator.pt");
#endif	
	std::cout<<"Generator numel:"<<generator->get_numel()<<std::endl;
	std::cout<<"Discriminator numel:"<<discriminator->get_numel()<<std::endl;
}
