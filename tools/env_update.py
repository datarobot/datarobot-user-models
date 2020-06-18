import argparse
import os

def main(dir_to_scan):
    """
    Build the public environment installer

    Parameters
    ----------
    release_version: str
        suffix to add to the end of the installer tarball to indicate its version
    envs_to_build: list
        list of environments to build; if None, all the public environments will be built
    """
    for item in os.listdir(dir_to_scan):
        print(item)


if __name__ == "__main__":
    CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.abspath(os.path.join(CUR_DIR, ".."))
    PUBLIC_ENVS_DIR_NAME = "public_dropin_environments"
    print(ROOT_DIR)
    #print(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Build the Custom Environment Installer")
    parser.add_argument(
        "-d",
        "--dir",
        default=os.path.join(ROOT_DIR, ),
        help="The DataRobot release the installer should be associated with. This is appended to "
        "the name of the final artifact. If ommitted, the value is taken from the "
        'DATAROBOT_RELEASE_VERSION environment variable or "test" if that is not set',
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="enable verbose output,")
    args = parser.parse_args()
    if args.verbose:
        print(args)
        IS_VERBOSE = True
    main(args.release_version, args.envs)
