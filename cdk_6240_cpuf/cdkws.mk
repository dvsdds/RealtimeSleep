.PHONY: clean All Project_Title Project_Build Project_PostBuild

All: Project_Title Project_Build Project_PostBuild

Project_Title:
	@echo "----------Building project:[ myproject - psdf ]----------"

Project_Build:
	@make -r -f myproject.mk -j 16 -C  ./ 

Project_PostBuild:
	@echo Executing Post Build commands ...
	@export CDKPath="D:/C-Sky/CDK" CDK_VERSION="V2.24.11" CPU="E906FP" ProjectName="myproject" ProjectPath="D:/realtime-sleep/RS6x_7x_mmWave_sdk_V2.0.2/Software_Kit/02_SDK/psdf_sdk/project/mmwave/myproject/cdk_6240_cpuf/" && D:/realtime-sleep/RS6x_7x_mmWave_sdk_V2.0.2/Software_Kit/02_SDK/psdf_sdk/project/mmwave/myproject/cdk_6240_cpuf/after_build.sh
	@echo Done


clean:
	@echo "----------Cleaning project:[ myproject - psdf ]----------"

