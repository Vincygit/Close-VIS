################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/Calibrator.cpp \
../src/ConfReader.cpp \
../src/INITest.cpp \
../src/PhotometricStereoSolver.cpp \
../src/Utils.cpp \
../src/main.cpp 

OBJS += \
./src/Calibrator.o \
./src/ConfReader.o \
./src/INITest.o \
./src/PhotometricStereoSolver.o \
./src/Utils.o \
./src/main.o 

CPP_DEPS += \
./src/Calibrator.d \
./src/ConfReader.d \
./src/INITest.d \
./src/PhotometricStereoSolver.d \
./src/Utils.d \
./src/main.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/usr/local/include/opencv2 -I/usr/include/boost -I/usr/local/include/opencv -O3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


