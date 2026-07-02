import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import config
from app.services import material


class TestMaterialTlsVerification(unittest.TestCase):
    def setUp(self):
        self.original_app_config = dict(config.app)
        self.original_proxy_config = dict(config.proxy)

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)
        config.proxy.clear()
        config.proxy.update(self.original_proxy_config)

    def test_search_pexels_uses_tls_verification_by_default(self):
        """
        默认路径必须开启 TLS 校验，避免素材 API key 和返回的素材 URL
        在公共网络或不可信代理环境中被中间人攻击截获或篡改。
        """
        config.app["pexels_api_keys"] = ["pexels-key"]
        config.app.pop("tls_verify", None)
        config.proxy.clear()

        fake_response = SimpleNamespace(
            json=lambda: {
                "videos": [
                    {
                        "duration": 8,
                        "video_files": [
                            {
                                "width": 1080,
                                "height": 1920,
                                "link": "https://example.com/video.mp4",
                            }
                        ],
                    }
                ]
            }
        )

        with patch("app.services.material.requests.get", return_value=fake_response) as get:
            results = material.search_videos_pexels("cat", minimum_duration=1)

        self.assertEqual(len(results), 1)
        self.assertTrue(get.call_args.kwargs["verify"])

    def test_search_pixabay_allows_explicit_tls_disable_for_proxy(self):
        """
        少数企业代理会使用自签证书。该场景必须显式配置关闭 TLS 校验，
        不能再由代码硬编码默认关闭。
        """
        config.app["pixabay_api_keys"] = ["pixabay-key"]
        config.app["tls_verify"] = False
        config.proxy.clear()

        fake_response = SimpleNamespace(
            json=lambda: {
                "hits": [
                    {
                        "duration": 8,
                        "videos": {
                            "large": {
                                "width": 1920,
                                "url": "https://example.com/video.mp4",
                            }
                        },
                    }
                ]
            }
        )

        with patch("app.services.material.requests.get", return_value=fake_response) as get:
            results = material.search_videos_pixabay("cat", minimum_duration=1)

        self.assertEqual(len(results), 1)
        self.assertFalse(get.call_args.kwargs["verify"])

    def test_save_video_uses_tls_verification_by_default(self):
        config.app.pop("tls_verify", None)
        config.proxy.clear()

        fake_response = SimpleNamespace(content=b"fake-video")

        class FakeVideoFileClip:
            duration = 1
            fps = 24

            def __init__(self, path):
                self.path = path

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "app.services.material.requests.get", return_value=fake_response
            ) as get, patch("app.services.material.VideoFileClip", FakeVideoFileClip):
                video_path = material.save_video(
                    "https://example.com/video.mp4?token=abc", save_dir=temp_dir
                )

            self.assertTrue(os.path.exists(video_path))
            self.assertTrue(get.call_args.kwargs["verify"])

    def test_download_videos_accepts_plain_string_concat_mode(self):
        """
        download_videos 可能被服务层或测试直接传入字符串模式，而不是
        VideoConcatMode 枚举。这里用空搜索词避免真实网络请求，只验证
        字符串 "random" 不会再因为访问 `.value` 抛 AttributeError。
        """
        result = material.download_videos(
            task_id="string-concat-mode",
            search_terms=[],
            video_concat_mode="random",
        )

        self.assertEqual(result, [])

    def test_download_videos_can_round_robin_terms_in_script_order(self):
        """
        开启按文案顺序匹配素材后，不能让第一个关键词的多个候选先把
        音频时长填满。这里模拟两个关键词各有多个候选，验证下载顺序是
        term1-第1个、term2-第1个、term1-第2个，贴近脚本叙事顺序。
        """
        search_results = {
            "opening city": [
                material.MaterialInfo(provider="pexels", url="https://v.example/a1.mp4", duration=3),
                material.MaterialInfo(provider="pexels", url="https://v.example/a2.mp4", duration=3),
            ],
            "middle office": [
                material.MaterialInfo(provider="pexels", url="https://v.example/b1.mp4", duration=3),
                material.MaterialInfo(provider="pexels", url="https://v.example/b2.mp4", duration=3),
            ],
        }
        downloaded_urls = []

        def fake_search(search_term, minimum_duration, video_aspect):
            return search_results[search_term]

        def fake_save_video(video_url, save_dir=""):
            downloaded_urls.append(video_url)
            return f"/tmp/{video_url.rsplit('/', 1)[-1]}"

        with (
            patch.dict(config.app, {"material_directory": ""}),
            patch.object(material, "search_videos_pexels", side_effect=fake_search),
            patch.object(material, "save_video", side_effect=fake_save_video),
        ):
            result = material.download_videos(
                task_id="ordered-materials",
                search_terms=["opening city", "middle office"],
                source="pexels",
                audio_duration=7,
                max_clip_duration=3,
                match_script_order=True,
            )

        self.assertEqual(
            downloaded_urls,
            [
                "https://v.example/a1.mp4",
                "https://v.example/b1.mp4",
                "https://v.example/a2.mp4",
            ],
        )
        self.assertEqual(result, ["/tmp/a1.mp4", "/tmp/b1.mp4", "/tmp/a2.mp4"])

    def test_download_videos_with_terms_maps_path_to_search_term(self):
        """
        download_videos_with_terms 必须保留“每个下载文件来自哪个搜索词”的映射，
        这是语义匹配 (video_semantic_match) 把素材和旁白内容对齐的输入。
        """
        config.app.pop("material_directory", None)
        config.proxy.clear()

        cat_item = material.MaterialInfo()
        cat_item.provider = "pexels"
        cat_item.url = "https://example.com/cat.mp4"
        cat_item.duration = 10

        money_item = material.MaterialInfo()
        money_item.provider = "pexels"
        money_item.url = "https://example.com/money.mp4"
        money_item.duration = 10

        def fake_search(search_term, minimum_duration, video_aspect):
            return {"cat": [cat_item], "money": [money_item]}.get(search_term, [])

        saved = {
            "https://example.com/cat.mp4": "/tmp/cat-saved.mp4",
            "https://example.com/money.mp4": "/tmp/money-saved.mp4",
        }

        with patch(
            "app.services.material.search_videos_pexels", side_effect=fake_search
        ), patch(
            "app.services.material.save_video",
            side_effect=lambda video_url, save_dir="": saved[video_url],
        ):
            paths, clip_terms = material.download_videos_with_terms(
                task_id="t-terms",
                search_terms=["cat", "money"],
                source="pexels",
                video_concat_mode="sequential",
                audio_duration=100,
                max_clip_duration=5,
            )

        self.assertEqual(set(paths), {"/tmp/cat-saved.mp4", "/tmp/money-saved.mp4"})
        self.assertEqual(clip_terms["/tmp/cat-saved.mp4"], "cat")
        self.assertEqual(clip_terms["/tmp/money-saved.mp4"], "money")

    def test_download_videos_wrapper_still_returns_plain_list(self):
        """旧调用方仍然只拿到路径列表，保持向后兼容。"""
        result = material.download_videos(
            task_id="wrapper-compat",
            search_terms=[],
            video_concat_mode="random",
        )
        self.assertIsInstance(result, list)


def _make_item(url: str, duration: int = 10) -> material.MaterialInfo:
    item = material.MaterialInfo()
    item.provider = "pexels"
    item.url = url
    item.duration = duration
    return item


class TestInteractiveSearch(unittest.TestCase):
    """
    download_videos_interactively: 生成关键词的同一个 LLM 参与搜索循环，
    根据每轮各搜索词命中的素材量继续改进搜索词，直到素材时长足够。
    """

    def setUp(self):
        self.original_app_config = dict(config.app)
        config.app.pop("material_directory", None)

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)

    def test_no_llm_call_when_first_round_is_enough(self):
        def fake_search(search_term, minimum_duration, video_aspect):
            return {
                "cat": [_make_item("https://e.com/cat1.mp4"), _make_item("https://e.com/cat2.mp4")]
            }.get(search_term, [])

        with patch(
            "app.services.material.search_videos_pexels", side_effect=fake_search
        ), patch(
            "app.services.material.save_video",
            side_effect=lambda video_url, save_dir="": f"/tmp/{video_url.rsplit('/', 1)[-1]}",
        ), patch(
            "app.services.llm.generate_terms_with_feedback"
        ) as llm_mock, patch(
            "app.services.llm.select_relevant_videos", return_value=None
        ):
            paths, clip_terms = material.download_videos_interactively(
                task_id="t-interactive",
                video_subject="cats",
                video_script="a script about cats",
                initial_terms=["cat"],
                video_concat_mode="sequential",
                audio_duration=10,  # 2 clips x 5s usable covers it
                max_clip_duration=5,
            )

        llm_mock.assert_not_called()
        self.assertEqual(len(paths), 2)
        self.assertTrue(all(term == "cat" for term in clip_terms.values()))

    def test_llm_refines_terms_until_duration_covered(self):
        def fake_search(search_term, minimum_duration, video_aspect):
            return {
                "cat": [_make_item("https://e.com/cat1.mp4"), _make_item("https://e.com/cat2.mp4")],
                "dog": [_make_item("https://e.com/dog1.mp4"), _make_item("https://e.com/dog2.mp4")],
            }.get(search_term, [])

        with patch(
            "app.services.material.search_videos_pexels", side_effect=fake_search
        ), patch(
            "app.services.material.save_video",
            side_effect=lambda video_url, save_dir="": f"/tmp/{video_url.rsplit('/', 1)[-1]}",
        ), patch(
            "app.services.llm.generate_terms_with_feedback",
            # "cat" is already tried and must be filtered; "dog" fills the gap.
            return_value=["cat", "dog"],
        ) as llm_mock, patch(
            "app.services.llm.select_relevant_videos", return_value=None
        ):
            paths, clip_terms = material.download_videos_interactively(
                task_id="t-interactive",
                video_subject="pets",
                video_script="a script about pets",
                initial_terms=["cat"],
                video_concat_mode="sequential",
                audio_duration=20,  # round 1 only yields 10s usable
                max_clip_duration=5,
            )

        llm_mock.assert_called_once()
        feedback = llm_mock.call_args.kwargs
        self.assertEqual(feedback["remaining_seconds"], 10)
        self.assertEqual(
            [entry["term"] for entry in feedback["search_history"]], ["cat"]
        )
        self.assertEqual(len(paths), 4)
        self.assertEqual(set(clip_terms.values()), {"cat", "dog"})

    def test_stops_when_llm_has_no_new_terms(self):
        def fake_search(search_term, minimum_duration, video_aspect):
            return {"cat": [_make_item("https://e.com/cat1.mp4")]}.get(search_term, [])

        with patch(
            "app.services.material.search_videos_pexels", side_effect=fake_search
        ), patch(
            "app.services.material.save_video",
            side_effect=lambda video_url, save_dir="": f"/tmp/{video_url.rsplit('/', 1)[-1]}",
        ), patch(
            "app.services.llm.generate_terms_with_feedback",
            return_value=["cat"],  # only repeats an already-tried term
        ) as llm_mock, patch(
            "app.services.llm.select_relevant_videos", return_value=None
        ):
            paths, _ = material.download_videos_interactively(
                task_id="t-interactive",
                video_subject="cats",
                video_script="a script about cats",
                initial_terms=["cat"],
                video_concat_mode="sequential",
                audio_duration=100,
                max_clip_duration=5,
            )

        # One refinement attempt, then the loop stops instead of spinning.
        llm_mock.assert_called_once()
        self.assertEqual(paths, ["/tmp/cat1.mp4"])


class TestTopicRelevanceFilter(unittest.TestCase):
    """
    filter_video_items_by_topic: 下载前由 LLM 根据素材标题/描述剔除跑题视频。
    LLM 不可用或全部拒绝时必须失败开放（保留全部素材），不能清空素材池。
    """

    def test_drops_videos_llm_marks_off_topic(self):
        items = [
            _make_item("https://e.com/cat.mp4"),
            _make_item("https://e.com/car.mp4"),
        ]
        items[0].description = "a cat playing"
        items[1].description = "a sports car driving"

        with patch(
            "app.services.llm.select_relevant_videos", return_value=[0]
        ) as llm_mock:
            kept = material.filter_video_items_by_topic(
                video_subject="cats",
                video_script="a script about cats",
                video_items=items,
                url_terms={"https://e.com/cat.mp4": "cat"},
            )

        self.assertEqual([item.url for item in kept], ["https://e.com/cat.mp4"])
        candidates = llm_mock.call_args.kwargs["candidates"]
        self.assertEqual(candidates[0]["description"], "a cat playing")
        self.assertEqual(candidates[0]["search_term"], "cat")

    def test_keeps_all_when_llm_unavailable(self):
        items = [_make_item("https://e.com/a.mp4"), _make_item("https://e.com/b.mp4")]
        with patch("app.services.llm.select_relevant_videos", return_value=None):
            kept = material.filter_video_items_by_topic(
                "cats", "script", items, url_terms={}
            )
        self.assertEqual(kept, items)

    def test_keeps_all_when_llm_rejects_everything(self):
        items = [_make_item("https://e.com/a.mp4")]
        with patch("app.services.llm.select_relevant_videos", return_value=[]):
            kept = material.filter_video_items_by_topic(
                "cats", "script", items, url_terms={}
            )
        self.assertEqual(kept, items)

    def test_interactive_download_skips_filtered_videos(self):
        def fake_search(search_term, minimum_duration, video_aspect):
            return {
                "cat": [
                    _make_item("https://e.com/cat1.mp4"),
                    _make_item("https://e.com/offtopic.mp4"),
                ]
            }.get(search_term, [])

        with patch(
            "app.services.material.search_videos_pexels", side_effect=fake_search
        ), patch(
            "app.services.material.save_video",
            side_effect=lambda video_url, save_dir="": f"/tmp/{video_url.rsplit('/', 1)[-1]}",
        ), patch(
            "app.services.llm.select_relevant_videos", return_value=[0]
        ):
            paths, _ = material.download_videos_interactively(
                task_id="t-filter",
                video_subject="cats",
                video_script="a script about cats",
                initial_terms=["cat"],
                video_contact_mode="sequential",
                audio_duration=5,
                max_clip_duration=5,
            )

        self.assertEqual(paths, ["/tmp/cat1.mp4"])


class TestProviderDescriptions(unittest.TestCase):
    def setUp(self):
        self.original_app_config = dict(config.app)

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)

    def test_pexels_page_title_from_slug(self):
        self.assertEqual(
            material._pexels_page_title(
                "https://www.pexels.com/video/a-woman-doing-yoga-855/"
            ),
            "a woman doing yoga",
        )
        self.assertEqual(material._pexels_page_title(""), "")

    def test_pixabay_search_captures_tags_as_description(self):
        config.app["pixabay_api_keys"] = ["pixabay-key"]
        fake_response = SimpleNamespace(
            json=lambda: {
                "hits": [
                    {
                        "duration": 8,
                        "tags": "cat, kitten, pet",
                        "videos": {
                            "large": {
                                "width": 1920,
                                "url": "https://example.com/video.mp4",
                            }
                        },
                    }
                ]
            }
        )
        with patch(
            "app.services.material.requests.get", return_value=fake_response
        ):
            results = material.search_videos_pixabay("cat", minimum_duration=1)
        self.assertEqual(results[0].description, "cat, kitten, pet")


class TestCoverrProvider(unittest.TestCase):
    """
    Coverr 视频素材源(spec: 2026-06-09-coverr-video-provider-design.md)。
    全部用 unittest.mock 替换 requests，确保 CI 不依赖真实网络和真实 API key。
    """

    def setUp(self):
        self.original_app_config = dict(config.app)
        self.original_proxy_config = dict(config.proxy)

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)
        config.proxy.clear()
        config.proxy.update(self.original_proxy_config)

    # ---------------- Tests for search_videos_coverr ----------------

    def test_search_coverr_uses_mp4_download_url(self):
        """
        search_videos_coverr 应把每个 hit 转成 MaterialInfo，并把 urls.mp4_download
        直接作为 MaterialInfo.url。
        按 Coverr 官方文档 (api.coverr.co/docs/videos/#download-a-video),
        GET mp4_download 本身就被 Coverr 计入下载统计,无需额外 PATCH ping。
        同时验证 Authorization header 使用 Bearer scheme。
        """
        config.app["coverr_api_keys"] = ["coverr-key"]
        config.app.pop("tls_verify", None)
        config.proxy.clear()

        fake_response = SimpleNamespace(
            json=lambda: {
                "page": 0,
                "pages": 50,
                "page_size": 20,
                "total": 1,
                "hits": [
                    {
                        "id": "S1YbPl1NfI",
                        "duration": 11.625,
                        "aspect_ratio": "16:9",
                        "urls": {
                            "mp4": "https://storage.coverr.co/videos/abc?token=xyz",
                            "mp4_preview": "https://storage.coverr.co/videos/abc/preview?token=xyz",
                            "mp4_download": "https://storage.coverr.co/videos/abc/download?token=xyz",
                        },
                    }
                ],
            }
        )

        with patch(
            "app.services.material.requests.get", return_value=fake_response
        ) as get:
            results = material.search_videos_coverr("nature", minimum_duration=5)

        self.assertEqual(len(results), 1)
        item = results[0]
        self.assertEqual(item.provider, "coverr")
        self.assertEqual(item.duration, 11)
        # url 字段就是 mp4_download URL,不再做 coverr://id|url 编码
        self.assertEqual(
            item.url, "https://storage.coverr.co/videos/abc/download?token=xyz"
        )
        # Bearer auth + TLS verify on by default
        self.assertEqual(
            get.call_args.kwargs["headers"]["Authorization"], "Bearer coverr-key"
        )
        self.assertTrue(get.call_args.kwargs["verify"])

    def test_search_coverr_uses_tls_verification_by_default(self):
        """与 pexels/pixabay 一致:未显式配置时 TLS 校验默认开启。"""
        config.app["coverr_api_keys"] = ["coverr-key"]
        config.app.pop("tls_verify", None)
        config.proxy.clear()

        fake_response = SimpleNamespace(json=lambda: {"hits": []})

        with patch(
            "app.services.material.requests.get", return_value=fake_response
        ) as get:
            material.search_videos_coverr("nature", minimum_duration=1)

        self.assertTrue(get.call_args.kwargs["verify"])

    def test_search_coverr_allows_explicit_tls_disable_for_proxy(self):
        """企业自签证书代理场景必须能显式关闭 TLS 校验。"""
        config.app["coverr_api_keys"] = ["coverr-key"]
        config.app["tls_verify"] = False
        config.proxy.clear()

        fake_response = SimpleNamespace(json=lambda: {"hits": []})

        with patch(
            "app.services.material.requests.get", return_value=fake_response
        ) as get:
            material.search_videos_coverr("nature", minimum_duration=1)

        self.assertFalse(get.call_args.kwargs["verify"])

    def test_search_coverr_filters_by_min_duration_and_accepts_string(self):
        """
        Coverr duration 字段在不同响应里可能是 number 或 string,
        两种格式都要接受;低于 minimum_duration 的应被过滤。
        """
        config.app["coverr_api_keys"] = ["coverr-key"]
        config.app.pop("tls_verify", None)
        config.proxy.clear()

        fake_response = SimpleNamespace(
            json=lambda: {
                "hits": [
                    {
                        "id": "shortvid",
                        "duration": 3,  # below minimum
                        "urls": {"mp4_download": "https://example.com/a.mp4"},
                    },
                    {
                        "id": "stringdur",
                        "duration": "10.500000",  # string accepted
                        "urls": {"mp4_download": "https://example.com/b.mp4"},
                    },
                ]
            }
        )

        with patch(
            "app.services.material.requests.get", return_value=fake_response
        ):
            results = material.search_videos_coverr("x", minimum_duration=5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].duration, 10)
        self.assertEqual(results[0].url, "https://example.com/b.mp4")

    def test_search_coverr_skips_invalid_items(self):
        """缺 id 或缺 urls.mp4_download 的条目应被跳过,不应抛异常。"""
        config.app["coverr_api_keys"] = ["coverr-key"]
        config.app.pop("tls_verify", None)
        config.proxy.clear()

        fake_response = SimpleNamespace(
            json=lambda: {
                "hits": [
                    {  # missing urls.mp4_download
                        "id": "no-download",
                        "duration": 10,
                        "urls": {"mp4_preview": "https://example.com/preview.mp4"},
                    },
                    {  # missing id
                        "duration": 10,
                        "urls": {"mp4_download": "https://example.com/x.mp4"},
                    },
                    {  # valid baseline
                        "id": "good",
                        "duration": 10,
                        "urls": {"mp4_download": "https://example.com/good.mp4"},
                    },
                ]
            }
        )

        with patch(
            "app.services.material.requests.get", return_value=fake_response
        ):
            results = material.search_videos_coverr("x", minimum_duration=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].url, "https://example.com/good.mp4")

    def test_search_coverr_returns_empty_on_failure(self):
        """
        响应结构异常 / 网络异常时,函数必须返回 [] 而不是抛异常,
        与 pexels/pixabay 行为保持一致。
        """
        config.app["coverr_api_keys"] = ["coverr-key"]
        config.app.pop("tls_verify", None)
        config.proxy.clear()

        # Subtest A: malformed response (no "hits" key)
        with self.subTest("malformed response"):
            fake_response = SimpleNamespace(
                json=lambda: {"error": "rate limited"}
            )
            with patch(
                "app.services.material.requests.get", return_value=fake_response
            ):
                results = material.search_videos_coverr("x", minimum_duration=1)
            self.assertEqual(results, [])

        # Subtest B: network exception bubbles up from requests.get
        with self.subTest("network exception"):
            with patch(
                "app.services.material.requests.get",
                side_effect=requests.ConnectionError("boom"),
            ):
                results = material.search_videos_coverr("x", minimum_duration=1)
            self.assertEqual(results, [])

    # ---------------- Tests for download_videos coverr branch ----------------

    def test_download_videos_passes_mp4_download_url_to_save_video(self):
        """
        在 source="coverr" 时:
          1. dispatch 到 search_videos_coverr
          2. coverr item 走通用下载路径:save_video 收到的就是 mp4_download URL
             (不再有 coverr://id|url 编码,也不再调用 PATCH ping)
          3. 返回保存路径
        """
        config.app["coverr_api_keys"] = ["coverr-key"]
        config.app.pop("tls_verify", None)
        config.app.pop("material_directory", None)
        config.proxy.clear()

        fake_item = material.MaterialInfo()
        fake_item.provider = "coverr"
        fake_item.url = "https://storage.coverr.co/videos/abc/download?token=xyz"
        fake_item.duration = 10

        with patch(
            "app.services.material.search_videos_coverr",
            return_value=[fake_item],
        ) as search, patch(
            "app.services.material.save_video",
            return_value="/tmp/coverr-saved.mp4",
        ) as save:
            result = material.download_videos(
                task_id="t-coverr",
                search_terms=["nature"],
                source="coverr",
                audio_duration=5,
                max_clip_duration=5,
            )

        # 1. dispatch
        self.assertEqual(search.call_count, 1)

        # 2. save_video 收到的就是 mp4_download URL,原样传入
        save_url = save.call_args.kwargs.get("video_url") or save.call_args.args[0]
        self.assertEqual(
            save_url, "https://storage.coverr.co/videos/abc/download?token=xyz"
        )

        # 3. 返回值正确
        self.assertEqual(result, ["/tmp/coverr-saved.mp4"])


if __name__ == "__main__":
    unittest.main()
