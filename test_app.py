import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import app


class ChatApiTests(unittest.TestCase):
    def test_chat_includes_previous_turns_in_model_request(self):
        captured = {}
        history = [
            {"role": "user", "content": "Hola"},
            {"role": "model", "content": "¡Hola! ¿En qué puedo ayudarte?"},
        ]

        def generate_content(**kwargs):
            captured["contents"] = kwargs["contents"]
            captured["config"] = kwargs["config"]
            return SimpleNamespace(text="It is a four-year program.")

        with patch.object(
            app, "client",
            SimpleNamespace(models=SimpleNamespace(
                generate_content=generate_content
            ))
        ), patch.object(app, "retrieve_documents", return_value=""):
            response = app.app.test_client().post(
                "/api/chat",
                data={
                    "message": "入学手続きについて教えてください。",
                    "history": json.dumps(history),
                }
            )

        self.assertEqual(response.status_code, 200)
        contents = captured["contents"]
        self.assertEqual(contents[0].role, "user")
        self.assertEqual(contents[0].parts[0].text, "Hola")
        self.assertEqual(contents[1].role, "model")
        self.assertEqual(
            contents[1].parts[0].text,
            "¡Hola! ¿En qué puedo ayudarte?"
        )
        self.assertIn(
            "User Question:\n入学手続きについて教えてください。",
            contents[2]
        )
        self.assertIn(
            "required response language for this turn is Japanese",
            contents[2]
        )
        self.assertIn(
            "required response language for this turn is Japanese",
            captured["config"].system_instruction
        )

    def test_detects_short_spanish_greeting(self):
        self.assertEqual(app._detect_language("Hola"), "Spanish")

    def test_defaults_to_english(self):
        self.assertEqual(app._response_language("", []), "English")

    def test_explicit_language_request_overrides_question_language(self):
        self.assertEqual(
            app._response_language("Please speak in Bangla: what is admission?", []),
            "Bengali"
        )

    def test_explicit_language_request_supports_direct_language_form(self):
        self.assertEqual(
            app._response_language("Answer me in French, please.", []),
            "French"
        )

    def test_detects_user_language_before_history(self):
        history = [{"role": "user", "content": "Hola"}]
        self.assertEqual(
            app._response_language("こんにちは", history),
            "Japanese"
        )

    def test_uses_current_question_language_not_history(self):
        history = [{"role": "user", "content": "Hola"}]
        self.assertEqual(
            app._response_language("ভর্তি কীভাবে হবে?", history),
            "Bengali"
        )

    def test_common_english_question_is_not_detected_as_swahili(self):
        self.assertEqual(
            app._detect_language("What is the admission process?"),
            "English"
        )

    def test_all_faculty_question_uses_room_for_complete_list(self):
        captured_config = {}

        def generate_content(**kwargs):
            captured_config["config"] = kwargs["config"]
            return SimpleNamespace(
                text="Prof. A — Professor.\nDr. B — Lecturer."
            )

        with patch.object(
            app, "client",
            SimpleNamespace(models=SimpleNamespace(
                generate_content=generate_content
            ))
        ), patch.object(
            app, "retrieve_documents",
            return_value="Source URL: https://isu.ac.bd/faculties/cse"
        ):
            response = app.app.test_client().post(
                "/api/chat",
                data={"message": "List all CSE faculty members"}
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("Dr. B", response.get_json()["response"])
        self.assertGreaterEqual(
            captured_config["config"].max_output_tokens,
            1000
        )

    def test_greeting_skips_website_retrieval_and_has_no_sources(self):
        fake_models = SimpleNamespace(
            generate_content=lambda **kwargs: SimpleNamespace(
                text="Hello! How can I help?"
            )
        )

        with patch.object(
            app, "client", SimpleNamespace(models=fake_models)
        ), patch.object(app, "retrieve_documents") as retrieve_documents:
            response = app.app.test_client().post(
                "/api/chat",
                data={"message": "Hi!"}
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["sources"], [])
        self.assertNotIn("grounding", response.get_json())
        retrieve_documents.assert_not_called()

    def test_chat_adds_matched_adjunct_category_if_model_omits_it(self):
        context = "\n".join([
            "Source URL: https://isu.ac.bd/faculties/department-of-computer-science-and-engineering",
            "Information:",
            "Page heading: Faculty Members (Department of Computer Science & Engineering)",
            "Faculty category: Adjunct Faculties",
            "Name: Md. Wahiduzzaman Wahid",
            "Personal title: Chief Technology Officer, Standard Group",
        ])
        fake_models = SimpleNamespace(
            generate_content=lambda **kwargs: SimpleNamespace(
                text=(
                    "Role: Chief Technology Officer, Standard Group. "
                    "Qualifications: BUET, DU, OCP."
                )
            )
        )

        with patch.object(
            app, "client", SimpleNamespace(models=fake_models)
        ), patch.object(
            app, "retrieve_documents", return_value=context
        ):
            response = app.app.test_client().post(
                "/api/chat",
                data={"message": "Tell me about Wahiduzzaman Wahid"}
            )

        self.assertEqual(response.status_code, 200)
        answer = response.get_json()["response"]
        self.assertIn("an adjunct faculty member", answer)
        self.assertIn("Department of Computer Science & Engineering", answer)


if __name__ == "__main__":
    unittest.main()
