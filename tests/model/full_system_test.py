#!/usr/bin/env python3

if __name__ == "__main__":
    """


    """

    import sys
    import os
    import time
    import json
    import requests
    from pathlib import Path
    from datetime import datetime

    project_root = Path(__file__).parent.parent.parent.parent  # tests/manual/model_testing -> 

    from src.utils.diagnostics import CrossPlatformDiagnostics
    from src.core.logging.global_logger import get_logger

    logger = get_logger("full_system_test")

    API_BASE_URL = "http://localhost:8000"
    TEST_RESULTS = {}


    def test_api_health():
        """API"""
        logger.info("=" * 60)
        logger.info("1: API")
        logger.info("=" * 60)

        try:
            response = requests.get(f"{API_BASE_URL}/api/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                logger.info(f" API: {data}")
                return True, data
            else:
                logger.error(f" API:  {response.status_code}")
                return False, None
        except Exception as e:
            logger.error(f" API: {e}")
            return False, None


    def test_api_info():
        """API"""
        logger.info("\n" + "=" * 60)
        logger.info("2: API")
        logger.info("=" * 60)

        try:
            response = requests.get(f"{API_BASE_URL}/api/info", timeout=10)
            if response.status_code == 200:
                data = response.json()
                logger.info(f" API")
                logger.info(f"  : {data.get('name', 'N/A')}")
                logger.info(f"  : {data.get('version', 'N/A')}")
                logger.info(f"  : {data.get('status', 'N/A')}")
                return True, data
            else:
                logger.error(f" API:  {response.status_code}")
                return False, None
        except Exception as e:
            logger.error(f" API: {e}")
            return False, None


    def test_monitoring_endpoints():
        """"""
        logger.info("\n" + "=" * 60)
        logger.info("3: ")
        logger.info("=" * 60)

        endpoints = ["/api/monitoring/status", "/api/monitoring/memory", "/api/monitoring/network"]

        results = {}
        for endpoint in endpoints:
            try:
                response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f" {endpoint}: ")
                    results[endpoint] = True
                else:
                    logger.warning(f" {endpoint}:  {response.status_code}")
                    results[endpoint] = False
            except Exception as e:
                logger.warning(f" {endpoint}:  - {e}")
                results[endpoint] = False

        return all(results.values()), results


    def test_diagnostics_integration():
        """"""
        logger.info("\n" + "=" * 60)
        logger.info("4: ")
        logger.info("=" * 60)

        try:
            # 
            device = CrossPlatformDiagnostics.get_device_info()
            logger.info(f" : {device}")

            # 
            snapshot = CrossPlatformDiagnostics.dump_memory_snapshot()
            logger.info(f" ")
            logger.info(f"  : {snapshot.get('platform', 'N/A')}")
            logger.info(f"  CPU: {snapshot.get('cpu_percent', 'N/A')}%")
            logger.info(f"  : {snapshot.get('ram_used_gb', 'N/A'):.2f} GB")

            # 
            is_high = CrossPlatformDiagnostics.check_memory_threshold(95.0)
            logger.info(f" : {'' if is_high else ''}")

            # 
            CrossPlatformDiagnostics.clear_cache()
            logger.info(f" ")

            return True, snapshot
        except Exception as e:
            logger.error(f" : {e}")
            return False, None


    def test_image_classification():
        """"""
        logger.info("\n" + "=" * 60)
        logger.info("5: ")
        logger.info("=" * 60)

        # 
        try:
            from PIL import Image
            import io

            # 
            img = Image.new("RGB", (224, 224), color="red")
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="PNG")
            img_bytes.seek(0)

            # 
            files = {"file": ("test.png", img_bytes, "image/png")}
            data = {"use_model": "false", "use_attributes": "true", "model_name": "default"}

            start_time = time.time()
            response = requests.post(f"{API_BASE_URL}/api/classify", files=files, data=data, timeout=60)
            elapsed_time = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                logger.info(f" ")
                logger.info(f"  : {elapsed_time:.2f}")
                logger.info(f"  : {result.get('role', 'N/A')}")
                logger.info(f"  : {result.get('similarity', 'N/A')}")
                return True, result
            else:
                logger.error(f" :  {response.status_code}")
                logger.error(f"  : {response.text}")
                return False, None
        except Exception as e:
            logger.error(f" : {e}")
            return False, None


    def test_batch_classification():
        """"""
        logger.info("\n" + "=" * 60)
        logger.info("6: ")
        logger.info("=" * 60)

        try:
            from PIL import Image
            import io

            # 
            files = []
            for i in range(3):
                img = Image.new("RGB", (224, 224), color=["red", "green", "blue"][i])
                img_bytes = io.BytesIO()
                img.save(img_bytes, format="PNG")
                img_bytes.seek(0)
                files.append(("files", (f"test_{i}.png", img_bytes, "image/png")))

            # 
            start_time = time.time()
            response = requests.post(
                f"{API_BASE_URL}/api/classify/batch",
                files=files,
                data={"model_name": "default"},
                timeout=120,
            )
            elapsed_time = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                results = result.get("results", [])
                logger.info(f" ")
                logger.info(f"  : {elapsed_time:.2f}")
                logger.info(f"  : {len(results)}")
                return True, result
            else:
                logger.error(f" :  {response.status_code}")
                return False, None
        except Exception as e:
            logger.error(f" : {e}")
            return False, None


    def test_performance():
        """"""
        logger.info("\n" + "=" * 60)
        logger.info("7: ")
        logger.info("=" * 60)

        try:
            from PIL import Image
            import io

            # 
            img = Image.new("RGB", (224, 224), color="red")
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="PNG")

            # 
            response_times = []
            for i in range(5):
                img_bytes.seek(0)
                files = {"file": ("test.png", img_bytes, "image/png")}
                data = {"use_model": "false", "use_attributes": "true", "model_name": "default"}

                start_time = time.time()
                response = requests.post(
                    f"{API_BASE_URL}/api/classify", files=files, data=data, timeout=60
                )
                elapsed_time = time.time() - start_time

                if response.status_code == 200:
                    response_times.append(elapsed_time)
                    logger.info(f"   {i+1}: {elapsed_time:.2f}")
                else:
                    logger.warning(f"   {i+1}: ")

            if response_times:
                avg_time = sum(response_times) / len(response_times)
                min_time = min(response_times)
                max_time = max(response_times)

                logger.info(f" ")
                logger.info(f"  : {avg_time:.2f}")
                logger.info(f"  : {min_time:.2f}")
                logger.info(f"  : {max_time:.2f}")

                return True, {"avg_time": avg_time, "min_time": min_time, "max_time": max_time}
            else:
                logger.error(f" : ")
                return False, None
        except Exception as e:
            logger.error(f" : {e}")
            return False, None


    def run_all_tests():
        """"""
        logger.info("\n" + "=" * 60)
        logger.info("")
        logger.info("=" * 60)
        logger.info(f": {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"API: {API_BASE_URL}")
        logger.info("")

        tests = [
            ("API", test_api_health),
            ("API", test_api_info),
            ("", test_monitoring_endpoints),
            ("", test_diagnostics_integration),
            ("", test_image_classification),
            ("", test_batch_classification),
            ("", test_performance),
        ]

        results = {}
        for test_name, test_func in tests:
            try:
                success, data = test_func()
                results[test_name] = {"success": success, "data": data}
            except Exception as e:
                logger.error(f" '{test_name}' : {e}")
                results[test_name] = {"success": False, "error": str(e)}

        # 
        logger.info("\n" + "=" * 60)
        logger.info("")
        logger.info("=" * 60)

        passed = 0
        failed = 0

        for test_name, result in results.items():
            status = " " if result["success"] else " "
            logger.info(f"{test_name}: {status}")
            if result["success"]:
                passed += 1
            else:
                failed += 1

        logger.info(f"\n: {passed}/{len(results)} ")

        if failed == 0:
            logger.info(" ")
        else:
            logger.warning(f" {failed} ")

        # 
        test_report = {
            "test_time": datetime.now().isoformat(),
            "api_url": API_BASE_URL,
            "results": results,
            "summary": {
                "total": len(results),
                "passed": passed,
                "failed": failed,
                "pass_rate": f"{passed/len(results)*100:.1f}%",
            },
        }

        report_file = project_root / "logs" / "full_system_test_report.json"
        report_file.parent.mkdir(exist_ok=True)
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(test_report, f, indent=2, ensure_ascii=False)

        logger.info(f"\n: {report_file}")

        return failed == 0


    if __name__ == "__main__":
        success = run_all_tests()
        sys.exit(0 if success else 1)
