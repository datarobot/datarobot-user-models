import datetime
import time
import os
import pprint
import json
import subprocess
from functools import partial
from multiprocessing import Lock, Manager, Pool, Process

from datarobot.mlops.connected.client import MLOpsClient
from datarobot.mlops.mlops import MLOps
import datarobot as dr

from datarobot_custom_code.utils import bytes_to_mb_str, calculate_rate, calculate_rate_str, parse_s3_uri, list_zip_contents, \
    extract_zip_content, load_yaml_from_file, sum_file_sizes, get_disk_space, merge_lists

from datarobot_custom_code.s3_file_download_helper import S3FileDownloadHelper


class SupportedStoreType:
    S3 = "s3"


class StoreURIPrefix:
    S3 = "s3:"


class ModelDownloader:
    REMOTE_FILE_SUFFIX = ".remote"
    METADATA_FILE = "model-metadata.yaml"
    MLOPS_RUNTIME_PARAM_PREFIX = "MLOPS_RUNTIME_PARAM_"

    def __init__(self,
                 deployment_id,
                 code_dir,
                 verify_ssl=True,
                 mb_to_update=50,
                 seconds_to_update=10,
                 chunk_size=1024*1024,
                 nr_processes=1,
                 verify_checksum=True):
        """
        ModelDownloader constructor
        :param deployment_id:  The deployment to download the model code from
        :param code_dir: The local code directory to use
        :param verify_ssl: If True verify SSL certificate
        """
        self._deployment_id = deployment_id
        self._dr_client = dr.Client()
        self._mlops_service_url = os.getenv("DATAROBOT_ENDPOINT")
        self._mlops_client = MLOpsClient(service_url=os.getenv("DATAROBOT_ENDPOINT").replace('/api/v2', ''),
                                         api_key=os.getenv("DATAROBOT_API_TOKEN"),
                                         verify=verify_ssl)
        self._mlops = MLOps().set_deployment_id(self._deployment_id).set_api_spooler().init()

        self._code_dir = code_dir
        self._deployment_info = self._mlops_client.get_deployment(deployment_id)
        self._model_package_id = self._deployment_info[MLOpsClient.RESPONSE_MODEL_PACKAGE_KEY][
            MLOpsClient.RESPONSE_MODEL_PACKAGE_ID_KEY
        ]
        pprint.pprint(self._deployment_info)
        print("model package id: {}".format(self._model_package_id))

        self._mb_to_update = mb_to_update
        self._seconds_to_update = seconds_to_update
        self._download_start_time = None
        self._current_downloaded_size = 0
        self._last_update_size = 0
        self._current_total_time_sec = 0
        self._chunk_size = chunk_size
        self._max_processes = nr_processes
        self._verify_checksum = verify_checksum

    @staticmethod
    def _handle_credentials_param(param_name):
        param_json = os.environ.get(param_name, None)
        if param_json is None:
            raise EnvironmentError("expected environment variable '{}' to be set".format(param_name))
        print(f"param_json: {param_json}")

        json_content = json.loads(param_json)
        if param_json is None:
            raise EnvironmentError("expected environment variable '{}' to be json".format(param_name))

        print("Successfully loaded JSON content:", json_content)
        payload = json_content["payload"]

        # For each cloud provider detect the credentials and set the env variables
        if "awsAccessKeyId" in payload:
            os.environ["AWS_ACCESS_KEY_ID"] = payload["awsAccessKeyId"]
        if "awsSecretAccessKey" in payload:
            os.environ["AWS_SECRET_ACCESS_KEY"] = payload["awsSecretAccessKey"]
        if "awsSessionToken" in payload:
            os.environ["AWS_SESSION_TOKEN"] = payload["awsSessionToken"]

    def _prepare_credentials(self):
        """
        Prepare the credentials of each ccloud provider according to the runtime parameters detected
        :return:
        """
        # Scan the metadata.yaml and detect the credentials type params
        # TODO move to a func...
        yaml_file_path = os.path.join(self._code_dir, self.METADATA_FILE)
        yaml_dict = load_yaml_from_file(yaml_file_path)
        if yaml_dict is None:
            raise Exception("Error reading: {}".format(yaml_file_path))
        if "runtimeParameterDefinitions" not in yaml_dict:
            raise Exception("Could not find runtimeParametersDefinitions in metadata file")

        runtime_params = yaml_dict["runtimeParameterDefinitions"]
        pprint.pprint(runtime_params)
        print("Scanning runtime params:")
        for runtime_parameter in runtime_params:
            print(runtime_parameter)
            if "type" not in runtime_parameter:
                raise Exception("Could not find type filed for runtimeParameter: {}".format(runtime_parameter))
            if runtime_parameter["type"] == "credential":
                param_name = runtime_parameter["fieldName"]
                print("Detected credentials param {}".format(param_name))
                self._handle_credentials_param(self.MLOPS_RUNTIME_PARAM_PREFIX + param_name)

    def _extract_model_dir(self, model_artifact_path, output_dir):
        print("extracting model to {}".format(output_dir))
        zip_content = list_zip_contents(model_artifact_path)
        print(zip_content)
        # detect the model zip inside
        model_zip = None
        for zip_filename in zip_content:
            if zip_filename.startswith("model") and zip_filename.endswith(".zip"):
                model_zip = zip_filename
        print("model zip filename: {}".format(model_zip))
        if model_zip is None:
            raise Exception("Error could not find model zip file: {}".format(zip_content))

        print("extracting")
        os.chdir(output_dir)

        extract_zip_content(model_artifact_path, file_to_extract=model_zip)
        model_zip_path = os.path.join(output_dir, model_zip)
        print(model_zip_path)
        if not os.path.exists(model_zip_path):
            raise Exception(
                "Error extracting model zip file - could not find model zip file: {}".format(model_zip_path))

        print("Extracting mode zip file: {}".format(model_zip_path))
        extract_zip_content(model_zip_path, dest_dir=self._code_dir)

    def download(self, output_dir, timeout=600, model_artifact_path=None):
        """
        Download the model code directory
        :param output_dir:
        :param timeout:
        :param model_artifact_path:
        :return:
        """
        if not model_artifact_path:
            model_artifact_path = self._mlops_client.download_model_package_from_registry(
                self._model_package_id,
                output_dir,
                download_scoring_code=False,
                scoring_code_binary=False,
                download_pps_installer=False,
                is_prediction_explanations_supported=False,
                timeout=timeout,
            )
        print("model artifact path: {}".format(model_artifact_path))
        self._extract_model_dir(model_artifact_path, output_dir)

    def _detect_remote_files(self):
        """
        Detect remote files in the code directory. A remote file is a file with a .remote suffix
        :return:
        """
        print("Scanning code dir for remote files:")
        print("-----------------------------------")
        remote_files = []
        for file in os.listdir(self._code_dir):
            print(file)
            if file.endswith(self.REMOTE_FILE_SUFFIX):
                print("Detected remote file")
                remote_files.append(os.path.join(self._code_dir, file))

        print("-----------------------------------")
        print("Found {} remote files".format(len(remote_files)))
        print(remote_files)
        print("+++++++++++++++++++++++++++++++++++\n")
        return remote_files

    @staticmethod
    def _get_remote_uri(file_path):
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
                if len(lines) == 1:
                    # Only one line containing the URI
                    uri = lines[0].strip()
                    checksum = None  # No checksum provided
                elif len(lines) == 2:
                    # Two lines, first containing the URI, second containing the checksum
                    uri = lines[0].strip()
                    checksum = lines[1].strip()
                    if checksum.startswith('sha256:'):
                        checksum = checksum[7:]
                    else:
                        raise Exception("Checksum line does not starts with sha256: [{}]".format(checksum))
                else:
                    raise ValueError("Invalid number of lines in the file")
            store_type = None
            if uri.startswith(StoreURIPrefix.S3):
                store_type = SupportedStoreType.S3
            return uri, store_type, checksum
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{file_path}' not found.")
        except Exception as e:
            raise e

    @staticmethod
    def _store_specific_info(remote_uri, store_type):
        if store_type == SupportedStoreType.S3:
            s3_helper = S3FileDownloadHelper()

            if s3_helper.is_uri_directory(remote_uri):
                is_dir = True
                file_size = 0
            else:
                file_size = s3_helper.get_file_size(remote_uri)
                is_dir = False
            return file_size, is_dir
        raise Exception("Unknown store type: {}".format(store_type))

    def _send_event(self, title, message):
        event_payload = {
            "eventType": "deploymentInfo",
            "title": title,
            "message": message,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "deploymentId": self._deployment_id,
        }
        remote_events_url = f"{self._mlops_service_url}/remoteEvents/"
        response = self._dr_client.post(
            url=remote_events_url,
            json=event_payload)
        response.raise_for_status()

    def _update_progress(self, size):
        """ We want to send an update every X GB downloaded or Y minutes passed """

        self._current_downloaded_size += size
        curr_size_mb = self._current_downloaded_size / (1024.0 * 1024.0)
        size_since_last_update_mb = (self._current_downloaded_size - self._last_update_size) / (1024.0 * 1024.0)

        if size_since_last_update_mb > self._mb_to_update:
            print("Updating progress... {:.1f} {:.1f}".format(size_since_last_update_mb, curr_size_mb))
            self._last_update_size = self._current_downloaded_size
            self._send_event(f"Downloaded {curr_size_mb} MB", "BlaBla")

    def _get_remote_dir_info(self, local_dir, dir_uri, store_type):

        if store_type == SupportedStoreType.S3:
            s3_helper = S3FileDownloadHelper()
            uri_list = s3_helper.list_uris_in_directory(dir_uri)
            uri_list.pop(0)
        else:
            raise Exception("Unsupported store type: {}".format(store_type))

        print("Files in {} directory:".format(dir_uri))
        print("=========================")
        print(uri_list)
        print("=========================")
        files_info = []
        for uri_info in uri_list:
            uri = uri_info["uri"]
            print("Processing uri: {}".format(uri))
            # Take the local file and use it as the base dir to the file inside the directory
            file_size, is_dir = self._store_specific_info(uri, store_type)
            print("File size: {}, is_dir: {}".format(file_size, is_dir))
            if is_dir:
                print(" ---> Not supporting a subdirectory for now")
            else:
                remote_path = uri_info["path"]
                print("local dir:   {}".format(local_dir))
                print("remote path: {} ".format(remote_path))
                print("parent dir:  {}".format(uri_info["parent_dir"]))
                print("code dir:    {}".format(self._code_dir))
                local_path = os.path.join(local_dir, remote_path.replace(uri_info["parent_dir"], "").lstrip("/"))
                print("local path: {}".format(local_path))
                files_info.append(self._build_file_info(remote_file=None,
                                                        local_file=local_path,
                                                        remote_uri=uri,
                                                        store_type=store_type,
                                                        file_size=file_size,
                                                        checksum=None))
        print("============")
        return files_info

    def _build_file_info(self, remote_file, local_file, remote_uri, store_type, file_size, checksum):
        return {
            "remote_file": remote_file,
            "local_file": local_file,
            "remote_uri": remote_uri,
            "store_type": store_type,
            "file_size": file_size,
            "file_size_mb": file_size / (1024.0 * 1024.0),
            "checksum": checksum,
        }

    def _get_remote_file_info(self, remote_file):
        local_file, extension = os.path.splitext(remote_file)
        remote_uri, store_type, checksum = self._get_remote_uri(remote_file)

        if remote_uri is None:
            raise Exception("No remote URI was detected in remote file: {}".format(remote_file))
        if store_type is None:
            raise Exception("Could not detect a supported store type for uri: {}".format(remote_uri))
        print("Remote URI: {}".format(remote_uri))
        print("Store type: {}".format(store_type))
        print("Checksum: [{}]".format(checksum))

        file_size, is_dir = self._store_specific_info(remote_uri, store_type)
        print("File size: {}".format(file_size))
        print("Is dir: {}".format(is_dir))
        if is_dir:
            print("Detected a directory for uri: {}".format(remote_uri))
            code_dir_path = os.path.dirname(local_file)
            destination_dir = os.environ.get("MODEL_STORE_PATH", code_dir_path)
            return self._get_remote_dir_info(destination_dir, remote_uri, store_type)
        else:
            print("No directory for uri: {}".format(remote_uri))
            return [self._build_file_info(remote_file, local_file, remote_uri, store_type, file_size, checksum)]

    def _build_remote_files_info(self, remote_files):
        remote_files_info = []
        for remote_file in remote_files:
            remote_files_info.extend(self._get_remote_file_info(remote_file))
        return remote_files_info

    def _parallel_remote_files_download(self, remote_files):

        # TODO: Small fixup... later move it into the function below
        for remote_file in remote_files:
            bucket_name, object_key = parse_s3_uri(remote_file["remote_uri"])
            remote_file["bucket_name"] = bucket_name
            remote_file["object_key"] = object_key

        manager = Manager()
        result_list = manager.list()
        lock = manager.Lock()
        for idx, d in enumerate(remote_files):
            d["index"] = idx
        print("Downloading files: {}".format(remote_files))
        print("Max processes: {}".format(self._max_processes))

        # pool = Pool(processes=self._max_processes)
        start_time_total = time.time()

        s3helper = S3FileDownloadHelper()
        partial_process = partial(s3helper.download_file)
        arg_list = [(result_list, file_info, self._code_dir, lock, self._chunk_size, self._verify_checksum)
                    for file_info in remote_files]

        processes = []
        for worker_args in arg_list:
            while len(processes) >= self._max_processes:
                time.sleep(0.5)  # Wait for some processes to finish
                for p in processes[:]:
                    if not p.is_alive():
                        p.join()
                        processes.remove(p)

            p = Process(target=partial_process, args=worker_args)
            processes.append(p)
            p.start()

        for p in processes:
            p.join()
        # pool.starmap(partial_process, arg_list)
        # pool.close()
        # pool.join()

        print("Received dictionaries from processes:", result_list)
        end_time_total = time.time()
        total_time = end_time_total - start_time_total
        total_size = sum_file_sizes(remote_files)

        print("Total file sizes: {}".format(bytes_to_mb_str(total_size)))
        print(f"Total time taken: {total_time:.2f} seconds")
        print("Overall bandwidth: {}".format(calculate_rate_str(total_size, total_time)))

        # Unifying files_to_downloads with result_list
        return merge_lists(remote_files, result_list)

    def _send_download_start_event(self, nr_files, total_size_mb):
        message = f"Total size: {total_size_mb:.1f} MB\n" + \
                  f"Number of processes: {self._max_processes}\n"
        self._send_event(title=f"Downloading {nr_files} remote files",
                         message=message)

    def prepare_download_summary(self, download_info, add_title=True, add_per_file_info=True):
        msg = ""
        if add_title:
            msg += "Done downloading remote files: \n"
        msg += "Total Time: {:.1f} sec\n".format(download_info["download_time"])
        msg += "Total size: {}\n".format(download_info["total_size_mb"])
        msg += "Rate:       {} MB/Sec\n".format(download_info["rate_mb_sec"])
        msg += "\nPer file info:\n"

        if add_per_file_info:
            for info in download_info["remote_files"]:
                msg += ("File: {} of size: {:.1f} MB, time: {:.1f} seconds, rate: {:.1f} MB/sec\n"
                        .format(info["local_file"], info["file_size_mb"],
                                info["total_time_sec"], info["rate_mb_sec"]))
        return msg

    def _send_download_end_event(self, download_info):
        msg = self.prepare_download_summary(download_info, add_title=False, add_per_file_info=True)
        self._send_event(title=f"Done downloading remote files",
                         message=msg)

    def _check_disk_space(self, total_size_mb):
        total_mb, used_mb, free_mb = get_disk_space(self._code_dir)
        print(f"Total disk space: {total_mb:.1f}MB, used disk space: {used_mb:.1f}, free disk space: {free_mb:}")
        if total_size_mb > free_mb:
            print("Error not enough disk space to download remote files: total_size_mb {} > {} free_mb"
                  .format(total_size_mb, free_mb))
            self._send_event("Error - not enough space to download remote files",
                             message=f"Total size of remote files: {total_size_mb:.1f}, free space: {free_mb:.1f}")
            raise Exception(f"Error not enough disk space to download: {total_size_mb} > {free_mb}")

    def _prepare_dir_structure(self, remote_files_info):
        for info in remote_files_info:
            local_file = info["local_file"]
            dir_name = os.path.dirname(local_file)
            os.makedirs(dir_name, exist_ok=True)

    def download_remote_files(self):
        """
        Download the remote files detected in the code directory
        :return:
        """
        print("---- Downloading remote files ----")
        # Calling prepare credentials here, as at this point we must have the model directory and if there are remote
        # files present, preparing the credentials is a must.
        self._prepare_credentials()

        # Scan code dir and get list of files
        remote_files = self._detect_remote_files()
        self._download_start_time = time.time()

        print("Getting remote files info")
        remote_files_info = self._build_remote_files_info(remote_files)
        print("----++++++")
        print(remote_files_info, flush=True)
        print("---------")
        total_size_mb = sum_file_sizes(remote_files_info) / (1024.0 * 1024.0)
        nr_files = len(remote_files_info)

        # At this point we have all the sizes of remote files (and a connection check was also done)
        # Checking that there is enough space to download all files
        self._check_disk_space(total_size_mb)

        self._prepare_dir_structure(remote_files_info)

        # Sending an event with info about the download that will start (number of files, total size)
        self._send_download_start_event(nr_files, total_size_mb)

        download_start_time = time.time()
        remote_files_info = self._parallel_remote_files_download(remote_files_info)
        download_time = time.time() - download_start_time

        download_info = {
            "total_size_mb": total_size_mb,
            "download_time": download_time,
            "rate_mb_sec": total_size_mb / download_time,
            "remote_files": remote_files_info
        }

        self._send_download_end_event(download_info)
        return download_info


def call_ls_on_directory(directory):
    try:
        # Run the ls command on the specified directory
        result = subprocess.run(['ls', "-lah", directory], capture_output=True, text=True, check=True)
        # If you are using Python 3.7 or earlier, use the following line instead:
        # result = subprocess.run(['ls', directory], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        # Extract and return the output of the ls command
        output = result.stdout
        return output
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return None


def drum_prepare(deployment_id,
                 local_dir="/tmp",
                 code_dir="/tmp/code",
                 skip_model_download=False,
                 model_artifact_path=None,
                 nr_processes=1):
    """
    An example of how to use the ModelDownloader class to prepare the code directory for DRUM to start
    :param deployment_id:
    :param local_dir:
    :param code_dir:
    :param skip_model_download: whether to skip the model download stage (for example if running inside init())
    :param model_artifact_path: If the model package is already downloaded .. save the download phase but open the
                model artifact
    :param nr_processes:
    :return:
    """
    md = ModelDownloader(deployment_id=deployment_id,
                         code_dir=code_dir,
                         chunk_size=1024*1024*10,
                         nr_processes=nr_processes)

    print("\n\n")
    start_time = time.time()
    print(f"Starting DRUM Prepare program")  # Press ⌘F8 to toggle the breakpoint.

    model_files_start_time = time.time()
    print(f"Downloading model files")
    print("model_artifact_path: ", model_artifact_path)
    if skip_model_download is False:
        md.download(output_dir=local_dir, model_artifact_path=model_artifact_path)
        print(f"Done downloading model files: {time.time() - model_files_start_time:.1f} sec")
    else:
        print("Skip downloading model files")

    print(f"Downloading remote files")
    download_info = md.download_remote_files()
    print(md.prepare_download_summary(download_info))

    end_time = time.time()
    print("\n\n\n")
    print(f"Total prep time: {end_time - start_time:.1f} sec")

    print("\n")
    print("--------- code dir after download ---------")
    output = call_ls_on_directory(code_dir)
    print(output)
