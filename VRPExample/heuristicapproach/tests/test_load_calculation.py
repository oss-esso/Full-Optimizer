import unittest
import sys
from pathlib import Path

# Add algo path to import Task
current_dir = Path(__file__).parent
heuristic_root = current_dir.parent
algo_dir = heuristic_root / 'algo'
sys.path.insert(0, str(algo_dir))

from epdt_data_structures import Task, TaskType

class TestLoadCalculation(unittest.TestCase):

    def get_load_change(self, task: Task):
        """
        This is a standalone copy of the corrected get_load_change function logic
        from tests/comprehensive_integration_test.py for isolated testing.
        """
        weight_change = getattr(task, 'demand', 0.0)
        volume_change = getattr(task, 'volume', 0.0)
        return weight_change, volume_change

    def test_pickup_task_load(self):
        """Verify load change for a PICKUP task."""
        pickup_task = Task(
            id='T1',
            location_id='LOC1',
            task_type=TaskType.PICKUP,
            order_id='ORDER1',
            lat=45.0,
            lon=9.0,
            service_time=15.0,
            demand=150.5,  # Positive for pickup
            volume=1.5     # Positive for pickup
        )
        weight, volume = self.get_load_change(pickup_task)
        self.assertEqual(weight, 150.5)
        self.assertEqual(volume, 1.5)

    def test_delivery_task_load(self):
        """Verify load change for a DELIVERY task."""
        delivery_task = Task(
            id='T2',
            location_id='LOC2',
            task_type=TaskType.DELIVERY,
            order_id='ORDER2',
            lat=45.0,
            lon=9.0,
            service_time=15.0,
            demand=-200.0, # Negative for delivery
            volume=-2.0    # Negative for delivery
        )
        weight, volume = self.get_load_change(delivery_task)
        self.assertEqual(weight, -200.0)
        self.assertEqual(volume, -2.0)

    def test_task_with_zero_load(self):
        """Verify behavior with zero load."""
        zero_load_task = Task(
            id='T3',
            location_id='LOC3',
            task_type=TaskType.PICKUP,
            order_id='ORDER3',
            lat=45.0,
            lon=9.0,
            service_time=15.0,
            demand=0,
            volume=0
        )
        weight, volume = self.get_load_change(zero_load_task)
        self.assertEqual(weight, 0)
        self.assertEqual(volume, 0)

    def test_task_missing_attributes(self):
        """Verify graceful failure if demand/volume are missing."""
        # Create a task object without demand/volume to test getattr default
        task_no_load = Task(
            id='T4',
            location_id='LOC4',
            task_type=TaskType.PICKUP,
            order_id='ORDER4',
            lat=45.0,
            lon=9.0,
            service_time=15.0
        )
        # Manually delete attributes to simulate missing data
        if hasattr(task_no_load, 'demand'):
            delattr(task_no_load, 'demand')
        if hasattr(task_no_load, 'volume'):
            delattr(task_no_load, 'volume')
            
        weight, volume = self.get_load_change(task_no_load)
        self.assertEqual(weight, 0.0)
        self.assertEqual(volume, 0.0)

if __name__ == '__main__':
    unittest.main()
