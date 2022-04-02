## 1) Oct 8, 2021 - Build Gantt Diagram Milestone 1
- Issue #31
- Build a Gantt chart with detailed deadlines for Milestone 1

## 2) Oct 8, 2021 - Define Features Milestone 2
- Issue #14
- Plan which Feature to include in the later stages

## 3) Oct 8, 2021 - Research how we can use transfer learning
- Issue #32
- Researched Nets that might be interesting for transfer learning and the loss functions we could use with them

## 4) Oct 8, 2021 - Prepare/define functional requirements for presentation
- Issue #32
- Discuss requirements in the group and prepare slides for presentation

## 4) Oct 13, 2021 - Prepare/define functional requirements for presentation
- Working on our presentation
- Come up with a timeplan and build a gantt chart out of it containing the relations
- Record voice for the requirements and gantt chart

## 5) 
- Work on the report writing detailed requirements section
- Research triplet loss function which is wiedly used in face recognition
- experiment with tensorflow and opencv 

## 6) Nov 21 - Build CNN Triplet Loss Skeleton and Prepare Peer Testing
- Come up with script to train CNN using triplet loss
- Build script to build triplet from folder structure on GPU server
- Prepare visualizations for peer testing

## 7) Nov 28 - Run Test on the GPU Server and implement Testing
- Set up environment on the GPU Server
- Run Experiments on the GPU server and adjust code
- Implement Testing
- Work on the Peer Testing Report


## 8) Feb 4 - Work on Gradient Explosion
- Double check the Network and the inputed data (found some issue there I could resolve)
- run further test, not yet working
- experiment with reduced problem to find out whether the problem comes from parameters or if it is a code issue

## 9) Feb 18 - Improve CNN 
- Train CNN in Siamese Similarity Setup on easy batch
- Improve performance with hard batch finding function
- research further improvement opportunities


## 10) Mar 4 - Research and Demo Video
- Further investigate Learning problems
- research medical image neural nets to use for transfer learning

## 11) Demo Video
- Create slides for CNN Demo
- Record Intro and CNN Video for Demo

## 12) Peer Report
- Add Overview section, describing the general purpose of the project and our approach
- Add detailed infos to the CNNs/Loss function and Siamese setup, we are using.

## 13) Mar 11 - Training on Simpler Dataset
- load in omingolet dataset in jpeg format in  anchor and positive folder structure to test the keras implementation
- training is currently in progress, report of results and insights follows asap
- also worked on the peer/team evalualtion

## 14) Mar 18 - Identified and fix GPU issue
- Finally found the big blocker and could resolve it with help of Quinn from BCCancer
- Run model after fixing to check functionality (looks like it works now)
- Look into which parameters are most promising to optimise

## 15) CNN Training issue fixed
- problem was an incompatability between the cuda drivers and the conda installation
- clean everything and reinstall the drivers and tensorflow using pip resolved the issue
- the test training on example data where the outcome is known, passed our tests, so we can finally continue

## 16) Train and Evaluate
- Train CNN with VGG Architecture in Siamese set up (two input mammograms)
- Evaluate model 
	- 87% accurarcy on 10 one shot task
	- clearly separable similarities for same/different patients mammograms fed
	
## 17) Prepare for Triplet Loss
- Start implementing the breast cropping algorithm into model
- Adjust triplet model to run on resetup gpu
- sort files for final training / parameter optimisation phase

## 18) Train Models VGG and ResNet
- Build architecture for ResNet
- Train ResNet and VGG
- Document training performance

## 19) Write Tests for the model
- Write test for the embedding network
- Write test for the Siamese Triplet set up

