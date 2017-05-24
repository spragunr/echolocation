# --------------------------------------------------------------
# Update the .bashrc file so that newly opened terminals will be
# configured correctly to work with a kinect-based turtlebot.
# --------------------------------------------------------------
if grep --quiet kinetic ~/.bashrc; then
    printf ".bashrc already configured.\n"
else
    printf "Configuring .bashrc\n"
  printf "\n" >> ~/.bashrc
  echo "export TURTLEBOT_3D_SENSOR=kinect" >> ~/.bashrc
  echo "source /opt/ros/kinetic/setup.bash" >> ~/.bashrc
  printf "\n" >> ~/.bashrc
  echo "if [ -e $HOME/catkin_ws/devel/setup.bash ]" >> ~/.bashrc
  echo "then" >> ~/.bashrc
  echo "  source $HOME/catkin_ws/devel/setup.bash" >> ~/.bashrc
  echo "fi" >> ~/.bashrc
fi

# --------------------------------------------------------------
# Source .bashrc so that *this* shell will be configured.
# --------------------------------------------------------------

source ~/.bashrc

# --------------------------------------------------------------
# Create and configure a catkin workspace, if necessary.
# --------------------------------------------------------------

WORKSPACE_SRC=$HOME"/catkin_ws/src/"

if [ ! -d $WORKSPACE_SRC ]; then
    printf "\nCreating catkin workspace.\n"
    mkdir -p $WORKSPACE_SRC
else
    printf "\nCatkin workspace already created.\n"
fi

if [ ! -f $WORKSPACE_SRC"CMakeLists.txt" ]; then
    printf "\nInitializing catkin workspace.\n"
    cd $WORKSPACE_SRC
    catkin_init_workspace
else
    printf "\nCatkin workspace already configured.\n"
fi
