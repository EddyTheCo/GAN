cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
include(local_conf.cmake OPTIONAL)
project(GANs VERSION 0.1 DESCRIPTION "Torch Generative Adversarial Network example" LANGUAGES CXX)


set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
	set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
		STRING "Choose the type of build." FORCE)
	# Set the possible values of build type for cmake-gui
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
		"Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_CXX_EXTENSIONS OFF)


add_executable(GAN GAN.cpp)
####################################CONFIG-VARIABLES#####################################
if(NOT  DISCRIMINATOR)
	set(DISCRIMINATOR "DGAN" CACHE STRING "DGAN MPS")
        message(STATUS "DISCRIMINATOR not set using default ${DISCRIMINATOR} from options DGAN MPS")
endif(NOT DISCRIMINATOR)
target_compile_definitions(GAN PRIVATE DISCRIMINATOR=${DISCRIMINATOR})
if(NOT  GENERATOR)
        set(GENERATOR "GGAN" CACHE STRING "GGAN")
        message(STATUS "GENERATOR not set using default ${GENERATOR} from options GGAN")
endif(NOT GENERATOR)
target_compile_definitions(GAN PRIVATE GENERATOR=${GENERATOR})

if(NOT DATASET)
	set(DATASET "MNIST" CACHE STRING "FP2 IRIS CMNIST MNIST ")
	message(STATUS "DATASET not set using default ${DATASET} from options IRIS CMNIST MNIST FP2")
endif(NOT DATASET)
target_compile_definitions(GAN PRIVATE DATASET=${DATASET})

option(TRAIN "On or off" ON)
option(TEST "On or off" OFF)
option(USE_YAML "On or off" ON)
#########################################################################################
if(TRAIN)
	target_compile_definitions(GAN PRIVATE TRAIN)
endif(TRAIN)
if(TEST)
	target_compile_definitions(GAN PRIVATE TEST)
	install(DIRECTORY  DESTINATION ./PICTURES)
endif(TEST)


include(FetchContent)
find_package(Torch REQUIRED)

target_link_libraries(GAN PRIVATE  "${TORCH_LIBRARIES}")
if(USE_YAML)
	if(NOT TARGET yaml-cpp)
		find_package(YAML-CPP CONFIG QUIET)
		if(NOT YAML-CPP_FOUND)
			message(STATUS "YAML-CPP was not found. Downloading")
			FetchContent_Declare(
				yaml-cpp
				GIT_REPOSITORY git@github.com:jbeder/yaml-cpp.git
				GIT_TAG yaml-cpp-0.7.0
				)
			FetchContent_MakeAvailable(yaml-cpp)
		else()
			message(STATUS "YAML FOUND in ${YAML-CPP_DIR}")
		endif(NOT YAML-CPP_FOUND)
		target_link_libraries(GAN PRIVATE  yaml-cpp)
	endif(NOT TARGET yaml-cpp)
endif(USE_YAML)

FetchContent_Declare(
	custom_modules
	GIT_REPOSITORY git@github.com:EddyTheCo/custom_modules.git
	GIT_TAG main
	)
FetchContent_MakeAvailable(custom_modules)


target_link_libraries(GAN PRIVATE  custom_modules)


FetchContent_Declare(
	custom_datasets
	GIT_REPOSITORY git@github.com:EddyTheCo/custom_datasets.git
	GIT_TAG main
	)
FetchContent_MakeAvailable(custom_datasets)

target_link_libraries(GAN PRIVATE  custom_datasets)

FetchContent_Declare(
	yaml-torch
	GIT_REPOSITORY git@github.com:EddyTheCo/yaml-torch.git
	GIT_TAG main
	)
FetchContent_MakeAvailable(yaml-torch)
target_link_libraries(GAN PRIVATE  yaml-torch)
if(TEST)
FetchContent_Declare(
	png-torch
	GIT_REPOSITORY git@github.com:EddyTheCo/png-torch.git
	GIT_TAG main
	)
FetchContent_MakeAvailable(png-torch)
target_link_libraries(GAN PRIVATE  png-torch)
endif(TEST)
target_compile_features(GAN PUBLIC cxx_std_17)



install(TARGETS GAN DESTINATION bin)
install(DIRECTORY INPUT DESTINATION .)
