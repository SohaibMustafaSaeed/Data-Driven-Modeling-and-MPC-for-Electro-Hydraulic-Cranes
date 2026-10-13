import threading

from .neural_controller import CraneNeuralController

_controller = None
_lock = threading.RLock()


def initialize_controller():
    """Load once; subsequent calls preserve recurrent memory."""
    global _controller
    with _lock:
        if _controller is None:
            _controller = CraneNeuralController()


def reset_controller(previous_command=None):
    with _lock:
        initialize_controller()
        _controller.reset(previous_command)


def controller_step(state22, previous_command=None, reference_01=None, reference_02=None):
    with _lock:
        initialize_controller()
        return _controller.step(state22, previous_command, reference_01, reference_02)
