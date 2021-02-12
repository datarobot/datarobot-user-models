from datarobot_drum.drum import docker_validation as dv


def test_valid_docker(good_docker):
    errors = [e for e in dv._check_for_errors(good_docker) if e is not None]
    assert len(errors) == 0


def test_broken_docker(broken_docker):
    errors = [e for e in dv._check_for_errors(broken_docker) if e is not None]
    assert len(errors) == 3
