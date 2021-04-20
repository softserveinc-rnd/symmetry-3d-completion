# Towards realistic symmetry-based completion of previously unseen point clouds
![](images/visualization.gif)

This is the anonymous demo code for the ICCV submission.


### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/ICCV2021Anonymous/symmetry_3D_completion.git
   ```
2. Install dependencies
   ```sh
   pip install -r requirements.txt
   ```


## Usage

As the proof of concept we added the visualization of each step so it is recommended to run the following command:
 ```sh
   python completion.py --filename=models/shapenet/plane.pcd --visualization=y --time=y
   ```

#### Flags
 * filepath  - path to the damaged pcd file to complete
 * visualization  - (y/n)  you can choose if you want the visualization of each step
 * time  -  (y/n) print time report

## Data

You can play with point clouds from different datasets:
* Damaged ShapeNet
* Damaged ModelNet
* Our new real-world dataset


Our full real-world dataset will be released with the camera-ready version.

