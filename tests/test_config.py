"""config のコード表。値は JV-Data仕様書_4.9.0.1.xlsx のコード表シート由来。"""

import pytest

from keiba import config


class TestTrackKind:
    @pytest.mark.parametrize(
        ("track_cd", "expected"),
        [
            ("10", "turf"),   # 平地 芝 直線
            ("11", "turf"),   # 平地 芝 左回り
            ("22", "turf"),   # 平地 芝 右回り 外2周
            ("23", "dirt"),   # 平地 ダート 左回り
            ("27", "dirt"),   # 平地 サンド 左回り
            ("29", "dirt"),   # 平地 ダート 直線
            ("51", "jump"),   # 障害 芝 襷
            ("59", "jump"),   # 障害 芝 外2周以上
            ("00", None),     # 未設定
            ("", None),
            ("99", None),
        ],
    )
    def test_maps_track_code(self, track_cd, expected):
        assert config.track_kind(track_cd) == expected


class TestClassLevel:
    @pytest.mark.parametrize(
        ("grade_cd", "jyoken", "expected"),
        [
            ("A", "999", 8),    # G1
            ("B", "999", 7),    # G2
            ("C", "999", 6),    # G3
            ("L", "999", 5),    # リステッド
            ("D", "999", 5),    # グレードのない重賞
            (" ", "701", 0),    # 新馬
            (" ", "702", 0),    # 未出走
            (" ", "703", 1),    # 未勝利
            (" ", "005", 2),    # 1勝クラス
            (" ", "010", 3),    # 2勝クラス
            (" ", "016", 4),    # 3勝クラス
            (" ", "999", 5),    # オープン
            ("E", "016", 4),    # 重賞以外の特別競走はクラスで決まる
            (" ", "001", 2),    # 旧100万下は1勝クラス相当
            (" ", "100", 5),    # 1600万超の条件は OP 相当
            (" ", "000", None), # 未設定は 0 にしない
            (" ", "", None),
        ],
    )
    def test_maps_class_level(self, grade_cd, jyoken, expected):
        assert config.class_level(grade_cd, jyoken) == expected


class TestCodeTables:
    def test_jyo_cd_covers_central_courses(self):
        assert config.JYO_CD_TO_NAME["01"] == "札幌"
        assert config.JYO_CD_TO_NAME["05"] == "東京"
        assert config.JYO_CD_TO_NAME["06"] == "中山"
        assert config.JYO_CD_TO_NAME["10"] == "小倉"
        assert len(config.JYO_CD_TO_NAME) == 10

    def test_tozai_cd_maps_to_training_centres(self):
        assert config.TOZAI_CD_TO_BASE["1"] == "美浦"
        assert config.TOZAI_CD_TO_BASE["2"] == "栗東"
        assert "0" not in config.TOZAI_CD_TO_BASE
        assert "3" not in config.TOZAI_CD_TO_BASE

    def test_transport_km_defined_for_every_base_and_course(self):
        for base in ("美浦", "栗東"):
            for jyo_cd in config.JYO_CD_TO_NAME:
                assert (base, jyo_cd) in config.TRANSPORT_KM

    def test_ijyo_groups_are_disjoint_and_keep_demotion_normal(self):
        # 6 落馬再騎乗 / 7 降着 は確定着順があるので通常扱い
        assert config.IJYO_EXCLUDED == frozenset({"1", "2", "3"})
        assert config.IJYO_NO_TIME == frozenset({"4", "5"})
        assert not (config.IJYO_EXCLUDED & config.IJYO_NO_TIME)
        assert "7" not in config.IJYO_EXCLUDED
        assert "7" not in config.IJYO_NO_TIME

    def test_data_kubun_priorities(self):
        assert config.DATA_KUBUN_RESULT_PRIORITY == ("7", "6", "5", "4", "3")
        assert config.DATA_KUBUN_ENTRY_PRIORITY == ("2", "1")
        assert config.DATA_KUBUN_DROP == frozenset({"0", "9", "A", "B"})


class TestVersions:
    """fix5 決定9-1: CSV の列構成と予想記録の形式は別の版で数える。"""

    def test_csv_and_record_schema_versions_are_separate_constants(self):
        assert isinstance(config.CSV_SCHEMA_VERSION, str)
        assert isinstance(config.RECORD_SCHEMA_VERSION, str)
        # 1つの SCHEMA_VERSION に戻すと、CSV の列を1つ足しただけで保存済みの
        # 予想記録が別形式を主張することになる（逆も同じ）。
        assert not hasattr(config, "SCHEMA_VERSION")

    def test_csv_schema_version_is_used_by_the_csv_writer(self):
        from keiba.ingest import store as ingest_store

        source = (ingest_store.__file__)
        with open(source, encoding="utf-8") as handle:
            text = handle.read()
        assert "config.CSV_SCHEMA_VERSION" in text
        assert "config.RECORD_SCHEMA_VERSION" not in text

    def test_record_schema_version_is_used_by_the_record_store(self):
        from keiba.store import records as records_store

        with open(records_store.__file__, encoding="utf-8") as handle:
            text = handle.read()
        assert "config.RECORD_SCHEMA_VERSION" in text
        assert "config.CSV_SCHEMA_VERSION" not in text


class TestTieBand:
    def test_the_comment_warns_against_applying_it_to_the_model_scale(self):
        """TIE_BAND の出典と「model 尺度に当ててはいけない」注意を残す（決定1）。

        この定数を model_win_prob に当てると、測っていない尺度のばらつきを
        主張し、スコアの幅が狭い run では上位n頭が消える。値だけを見て
        使い回されないよう、注意そのものをソースに置く。
        """
        with open(config.__file__, encoding="utf-8") as handle:
            text = handle.read()
        assert "当ててはいけない" in text
        assert "測り直す" in text


class TestVersionValues:
    """fix6 ①: 版番号が実態を主張していること。"""

    def test_record_schema_version_is_three(self):
        """記録の形式が2箇所増えたので "3"。

        "2" は usage に3フィールド（layer1_requests / layer2_requests /
        requests）が増えた世代で、その記録が data/predictions に実在する。
        "3" で増えたのは (a) ``data_quality.calibration``（較正を適用したか、
        していないならなぜか、再正規化前の合計とずれ）と (b)
        ``calibrated_win_prob`` が ``None`` とは限らなくなったこと。
        読む側の期待が変わるので、版を分けて記録自身から判別できるようにする。
        """
        assert config.RECORD_SCHEMA_VERSION == "3"

    def test_feature_version_is_three(self):
        """state の中身が2箇所変わったので、特徴量の版は "3"。

        1. 偽の「登録後に斤量・騎手が変更されている」注記が消える
           （``EntryRow.changed_after_entry`` のゼロ埋め判定。実データでは
           1,358頭中 1,346頭が偽の注記を持っていた）。
        2. ``data_quality.missing`` が「照合できたか」ではなく「引けた過去走が
           0本か」で立つようになり、欠損を申告する馬が増える。

        どちらも**モデルが見る入力そのもの**が変わるので、"2" の記録と
        答えを直接比べてはいけない。fix4 で horse_notes を新設したときに
        据え置いたのは state の中身が変わらなかったからで、今回は違う。
        """
        assert config.FEATURE_VERSION == "3"
