#!/bin/bash

echo "Generating common bash profile..."
{
    echo "#!/bin/bash"
    echo "# Setting user process limits."
    echo "ulimit -Su 2048"
    echo "ulimit -Hu 2048"
} > /etc/profile.d/bash-profile-load.sh
