#!/usr/bin/env python3
"""
cuOP Performance Dashboard

A comprehensive performance monitoring and analysis dashboard for cuOP.
Provides real-time monitoring, historical analysis, and optimization recommendations.
"""

import os
import sys
import json
import time
import threading
import queue
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging

# Web framework imports
try:
    from flask import Flask, render_template, jsonify, request, send_file
    from flask_socketio import SocketIO, emit
    import plotly.graph_objs as go
    import plotly.utils
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"Missing required dependencies: {e}")
    print("Please install: pip install flask flask-socketio plotly pandas numpy")
    sys.exit(1)

# cuOP imports
try:
    import cuop
    from cuop import PerformanceMonitor, MemoryAnalyzer, HotspotAnalyzer, AutoTuner
except ImportError:
    print("cuOP not found. Please ensure cuOP is properly installed.")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceDashboard:
    """Main performance dashboard class."""
    
    def __init__(self, host='0.0.0.0', port=5000, debug=False):
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'cuop_performance_dashboard_secret'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        self.host = host
        self.port = port
        self.debug = debug
        
        # Performance data storage
        self.performance_data = {
            'events': [],
            'memory_stats': [],
            'hotspots': [],
            'tuning_results': []
        }
        
        # Real-time monitoring
        self.monitoring_enabled = False
        self.monitoring_thread = None
        self.data_queue = queue.Queue()
        
        # Setup routes and event handlers
        self._setup_routes()
        self._setup_socketio_events()
        
        # Initialize cuOP performance monitoring
        self._initialize_cuop_monitoring()
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            return render_template('dashboard.html')
        
        @self.app.route('/api/performance/overview')
        def performance_overview():
            """Get performance overview data."""
            try:
                overview = self._get_performance_overview()
                return jsonify(overview)
            except Exception as e:
                logger.error(f"Error getting performance overview: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/performance/events')
        def performance_events():
            """Get performance events data."""
            try:
                events = self._get_performance_events()
                return jsonify(events)
            except Exception as e:
                logger.error(f"Error getting performance events: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/memory/stats')
        def memory_stats():
            """Get memory statistics."""
            try:
                stats = self._get_memory_stats()
                return jsonify(stats)
            except Exception as e:
                logger.error(f"Error getting memory stats: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/hotspots/analysis')
        def hotspots_analysis():
            """Get hotspot analysis data."""
            try:
                hotspots = self._get_hotspots_analysis()
                return jsonify(hotspots)
            except Exception as e:
                logger.error(f"Error getting hotspots analysis: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/tuning/results')
        def tuning_results():
            """Get auto-tuning results."""
            try:
                results = self._get_tuning_results()
                return jsonify(results)
            except Exception as e:
                logger.error(f"Error getting tuning results: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/monitoring/start', methods=['POST'])
        def start_monitoring():
            """Start real-time monitoring."""
            try:
                self.start_monitoring()
                return jsonify({'status': 'success', 'message': 'Monitoring started'})
            except Exception as e:
                logger.error(f"Error starting monitoring: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/monitoring/stop', methods=['POST'])
        def stop_monitoring():
            """Stop real-time monitoring."""
            try:
                self.stop_monitoring()
                return jsonify({'status': 'success', 'message': 'Monitoring stopped'})
            except Exception as e:
                logger.error(f"Error stopping monitoring: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/reports/generate', methods=['POST'])
        def generate_report():
            """Generate performance report."""
            try:
                report_data = request.get_json()
                report = self._generate_performance_report(report_data)
                return jsonify(report)
            except Exception as e:
                logger.error(f"Error generating report: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/export/csv')
        def export_csv():
            """Export performance data to CSV."""
            try:
                data_type = request.args.get('type', 'all')
                csv_data = self._export_to_csv(data_type)
                return send_file(csv_data, as_attachment=True, download_name=f'cuop_performance_{data_type}.csv')
            except Exception as e:
                logger.error(f"Error exporting CSV: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/export/json')
        def export_json():
            """Export performance data to JSON."""
            try:
                data_type = request.args.get('type', 'all')
                json_data = self._export_to_json(data_type)
                return send_file(json_data, as_attachment=True, download_name=f'cuop_performance_{data_type}.json')
            except Exception as e:
                logger.error(f"Error exporting JSON: {e}")
                return jsonify({'error': str(e)}), 500
    
    def _setup_socketio_events(self):
        """Setup SocketIO event handlers."""
        
        @self.socketio.on('connect')
        def handle_connect():
            logger.info('Client connected to dashboard')
            emit('status', {'message': 'Connected to cuOP Performance Dashboard'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            logger.info('Client disconnected from dashboard')
        
        @self.socketio.on('request_data')
        def handle_data_request(data):
            """Handle real-time data requests."""
            try:
                data_type = data.get('type', 'overview')
                response_data = self._get_realtime_data(data_type)
                emit('data_update', response_data)
            except Exception as e:
                logger.error(f"Error handling data request: {e}")
                emit('error', {'message': str(e)})
    
    def _initialize_cuop_monitoring(self):
        """Initialize cuOP performance monitoring."""
        try:
            # Initialize performance monitor
            self.performance_monitor = cuop.PerformanceMonitor()
            self.performance_monitor.StartMonitoring()
            
            # Initialize memory analyzer
            self.memory_analyzer = cuop.MemoryAnalyzer()
            self.memory_analyzer.StartTracking()
            
            # Initialize hotspot analyzer
            self.hotspot_analyzer = cuop.HotspotAnalyzer()
            self.hotspot_analyzer.StartAnalysis()
            
            # Initialize auto-tuner
            self.auto_tuner = cuop.AutoTuner()
            
            logger.info("cuOP performance monitoring initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize cuOP monitoring: {e}")
            raise
    
    def _get_performance_overview(self) -> Dict[str, Any]:
        """Get performance overview data."""
        try:
            # Get performance analysis
            analysis = self.performance_monitor.AnalyzePerformance()
            
            # Get memory stats
            memory_stats = self.memory_analyzer.GetMemoryStats()
            
            # Get hotspot analysis
            hotspots = self.hotspot_analyzer.AnalyzeHotspots()
            
            overview = {
                'timestamp': datetime.now().isoformat(),
                'total_execution_time': analysis.total_execution_time,
                'total_events': len(analysis.events),
                'gpu_utilization': analysis.gpu_utilization,
                'memory_bandwidth': analysis.memory_bandwidth,
                'memory_stats': {
                    'total_allocated': memory_stats.total_allocated,
                    'total_freed': memory_stats.total_freed,
                    'current_usage': memory_stats.current_usage,
                    'peak_usage': memory_stats.peak_usage,
                    'allocation_count': memory_stats.allocation_count
                },
                'top_hotspots': [
                    {
                        'function_name': hotspot.function_name,
                        'total_time': hotspot.total_time,
                        'percentage': hotspot.percentage,
                        'call_count': hotspot.call_count,
                        'avg_time': hotspot.avg_time
                    }
                    for hotspot in hotspots[:5]
                ],
                'recommendations': analysis.recommendations
            }
            
            return overview
            
        except Exception as e:
            logger.error(f"Error getting performance overview: {e}")
            return {'error': str(e)}
    
    def _get_performance_events(self) -> List[Dict[str, Any]]:
        """Get performance events data."""
        try:
            analysis = self.performance_monitor.AnalyzePerformance()
            
            events = []
            for event in analysis.events:
                events.append({
                    'name': event.name,
                    'type': event.type,
                    'duration': event.GetDuration(),
                    'memory_used': event.memory_used,
                    'memory_allocated': event.memory_allocated,
                    'device_info': event.device_info,
                    'kernel_info': event.kernel_info
                })
            
            return events
            
        except Exception as e:
            logger.error(f"Error getting performance events: {e}")
            return []
    
    def _get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        try:
            memory_stats = self.memory_analyzer.GetMemoryStats()
            
            return {
                'total_allocated': memory_stats.total_allocated,
                'total_freed': memory_stats.total_freed,
                'current_usage': memory_stats.current_usage,
                'peak_usage': memory_stats.peak_usage,
                'allocation_count': memory_stats.allocation_count,
                'free_count': memory_stats.free_count,
                'fragmentation_ratio': self.memory_analyzer.CalculateFragmentationRatio()
            }
            
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {}
    
    def _get_hotspots_analysis(self) -> List[Dict[str, Any]]:
        """Get hotspot analysis data."""
        try:
            hotspots = self.hotspot_analyzer.AnalyzeHotspots()
            
            hotspot_data = []
            for hotspot in hotspots:
                hotspot_data.append({
                    'function_name': hotspot.function_name,
                    'file_name': hotspot.file_name,
                    'line_number': hotspot.line_number,
                    'total_time': hotspot.total_time,
                    'percentage': hotspot.percentage,
                    'call_count': hotspot.call_count,
                    'avg_time': hotspot.avg_time,
                    'min_time': hotspot.min_time,
                    'max_time': hotspot.max_time,
                    'std_deviation': hotspot.std_deviation,
                    'bottleneck_type': hotspot.bottleneck_type,
                    'optimization_suggestion': hotspot.optimization_suggestion,
                    'priority_score': hotspot.priority_score
                })
            
            return hotspot_data
            
        except Exception as e:
            logger.error(f"Error getting hotspots analysis: {e}")
            return []
    
    def _get_tuning_results(self) -> List[Dict[str, Any]]:
        """Get auto-tuning results."""
        try:
            tuning_history = self.auto_tuner.GetTuningHistory()
            
            results = []
            for result in tuning_history:
                results.append({
                    'operator_name': result.operator_name,
                    'best_performance': result.best_performance,
                    'iterations_used': result.iterations_used,
                    'converged': result.converged,
                    'convergence_rate': result.convergence_rate,
                    'best_parameters': result.best_parameters,
                    'performance_history': result.performance_history,
                    'recommendations': result.recommendations
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting tuning results: {e}")
            return []
    
    def _get_realtime_data(self, data_type: str) -> Dict[str, Any]:
        """Get real-time data for WebSocket updates."""
        try:
            if data_type == 'overview':
                return self._get_performance_overview()
            elif data_type == 'events':
                return {'events': self._get_performance_events()}
            elif data_type == 'memory':
                return {'memory': self._get_memory_stats()}
            elif data_type == 'hotspots':
                return {'hotspots': self._get_hotspots_analysis()}
            else:
                return {'error': f'Unknown data type: {data_type}'}
                
        except Exception as e:
            logger.error(f"Error getting real-time data: {e}")
            return {'error': str(e)}
    
    def _generate_performance_report(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        try:
            report_type = report_data.get('type', 'comprehensive')
            include_charts = report_data.get('include_charts', True)
            
            # Get all performance data
            overview = self._get_performance_overview()
            events = self._get_performance_events()
            memory_stats = self._get_memory_stats()
            hotspots = self._get_hotspots_analysis()
            tuning_results = self._get_tuning_results()
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'report_type': report_type,
                'overview': overview,
                'events': events,
                'memory_stats': memory_stats,
                'hotspots': hotspots,
                'tuning_results': tuning_results,
                'summary': self._generate_report_summary(overview, hotspots, tuning_results)
            }
            
            if include_charts:
                report['charts'] = self._generate_chart_data(overview, events, memory_stats, hotspots)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {'error': str(e)}
    
    def _generate_report_summary(self, overview: Dict, hotspots: List[Dict], tuning_results: List[Dict]) -> Dict[str, Any]:
        """Generate report summary."""
        summary = {
            'total_execution_time': overview.get('total_execution_time', 0),
            'total_events': overview.get('total_events', 0),
            'memory_efficiency': self._calculate_memory_efficiency(overview.get('memory_stats', {})),
            'performance_score': self._calculate_performance_score(overview, hotspots),
            'optimization_opportunities': len([h for h in hotspots if h.get('priority_score', 0) > 0.7]),
            'tuning_success_rate': self._calculate_tuning_success_rate(tuning_results)
        }
        
        return summary
    
    def _calculate_memory_efficiency(self, memory_stats: Dict) -> float:
        """Calculate memory efficiency score."""
        try:
            if memory_stats.get('total_allocated', 0) == 0:
                return 1.0
            
            efficiency = 1.0 - (memory_stats.get('current_usage', 0) / memory_stats.get('total_allocated', 1))
            return max(0.0, min(1.0, efficiency))
        except:
            return 0.0
    
    def _calculate_performance_score(self, overview: Dict, hotspots: List[Dict]) -> float:
        """Calculate overall performance score."""
        try:
            score = 1.0
            
            # Penalize high execution time
            if overview.get('total_execution_time', 0) > 1000:  # 1 second
                score -= 0.2
            
            # Penalize high hotspot percentages
            for hotspot in hotspots[:3]:  # Top 3 hotspots
                if hotspot.get('percentage', 0) > 20:
                    score -= 0.1
            
            return max(0.0, min(1.0, score))
        except:
            return 0.5
    
    def _calculate_tuning_success_rate(self, tuning_results: List[Dict]) -> float:
        """Calculate auto-tuning success rate."""
        try:
            if not tuning_results:
                return 0.0
            
            successful_tunings = sum(1 for result in tuning_results if result.get('converged', False))
            return successful_tunings / len(tuning_results)
        except:
            return 0.0
    
    def _generate_chart_data(self, overview: Dict, events: List[Dict], memory_stats: Dict, hotspots: List[Dict]) -> Dict[str, Any]:
        """Generate chart data for visualization."""
        try:
            charts = {}
            
            # Performance events timeline
            if events:
                event_times = [event.get('duration', 0) for event in events]
                event_names = [event.get('name', 'Unknown') for event in events]
                
                charts['events_timeline'] = {
                    'x': event_names,
                    'y': event_times,
                    'type': 'bar',
                    'title': 'Performance Events Timeline'
                }
            
            # Memory usage over time
            if memory_stats:
                charts['memory_usage'] = {
                    'labels': ['Total Allocated', 'Current Usage', 'Peak Usage'],
                    'values': [
                        memory_stats.get('total_allocated', 0),
                        memory_stats.get('current_usage', 0),
                        memory_stats.get('peak_usage', 0)
                    ],
                    'type': 'pie',
                    'title': 'Memory Usage Distribution'
                }
            
            # Top hotspots
            if hotspots:
                top_hotspots = hotspots[:10]
                hotspot_names = [h.get('function_name', 'Unknown') for h in top_hotspots]
                hotspot_times = [h.get('total_time', 0) for h in top_hotspots]
                
                charts['top_hotspots'] = {
                    'x': hotspot_names,
                    'y': hotspot_times,
                    'type': 'bar',
                    'title': 'Top Performance Hotspots'
                }
            
            return charts
            
        except Exception as e:
            logger.error(f"Error generating chart data: {e}")
            return {}
    
    def _export_to_csv(self, data_type: str) -> str:
        """Export performance data to CSV."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"/tmp/cuop_performance_{data_type}_{timestamp}.csv"
            
            if data_type == 'events':
                events = self._get_performance_events()
                df = pd.DataFrame(events)
            elif data_type == 'hotspots':
                hotspots = self._get_hotspots_analysis()
                df = pd.DataFrame(hotspots)
            elif data_type == 'memory':
                memory_stats = self._get_memory_stats()
                df = pd.DataFrame([memory_stats])
            else:
                # Export all data
                overview = self._get_performance_overview()
                df = pd.DataFrame([overview])
            
            df.to_csv(filename, index=False)
            return filename
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            raise
    
    def _export_to_json(self, data_type: str) -> str:
        """Export performance data to JSON."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"/tmp/cuop_performance_{data_type}_{timestamp}.json"
            
            if data_type == 'events':
                data = self._get_performance_events()
            elif data_type == 'hotspots':
                data = self._get_hotspots_analysis()
            elif data_type == 'memory':
                data = self._get_memory_stats()
            else:
                # Export all data
                data = self._generate_performance_report({'type': 'comprehensive'})
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            return filename
            
        except Exception as e:
            logger.error(f"Error exporting to JSON: {e}")
            raise
    
    def start_monitoring(self):
        """Start real-time monitoring."""
        if self.monitoring_enabled:
            logger.warning("Monitoring is already enabled")
            return
        
        self.monitoring_enabled = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Real-time monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        if not self.monitoring_enabled:
            logger.warning("Monitoring is not enabled")
            return
        
        self.monitoring_enabled = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        logger.info("Real-time monitoring stopped")
    
    def _monitoring_loop(self):
        """Real-time monitoring loop."""
        while self.monitoring_enabled:
            try:
                # Collect performance data
                overview = self._get_performance_overview()
                
                # Emit real-time updates
                self.socketio.emit('performance_update', overview)
                
                # Wait before next update
                time.sleep(1.0)  # Update every second
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5.0)  # Wait longer on error
    
    def run(self):
        """Run the performance dashboard."""
        logger.info(f"Starting cuOP Performance Dashboard on {self.host}:{self.port}")
        
        try:
            self.socketio.run(self.app, host=self.host, port=self.port, debug=self.debug)
        except KeyboardInterrupt:
            logger.info("Dashboard stopped by user")
        except Exception as e:
            logger.error(f"Error running dashboard: {e}")
        finally:
            self.stop_monitoring()
            logger.info("Dashboard shutdown complete")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='cuOP Performance Dashboard')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Create and run dashboard
    dashboard = PerformanceDashboard(host=args.host, port=args.port, debug=args.debug)
    dashboard.run()

if __name__ == '__main__':
    main()
