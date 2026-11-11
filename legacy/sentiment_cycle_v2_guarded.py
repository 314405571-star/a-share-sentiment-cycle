import baostock as bs
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
import os
import pickle
import traceback
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('emotion_cycle.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EmotionCycleAnalyzer:
    """
    情绪周期日度分析器（行业版，增强行业识别与日期校验）
    """

    def __init__(self, csv_path: str, code_column: str = 'code',
                 name_column: str = 'name', state_file: str = 'emotion_state.pkl',
                 max_retries: int = 3, request_delay: float = 0.3,
                 max_workers: int = 2):
        self.csv_path = csv_path
        self.code_column = code_column
        self.name_column = name_column
        self.state_file = state_file
        self.max_retries = max_retries
        self.request_delay = request_delay
        self.max_workers = max_workers
        self.is_logged_in = False

        self.stock_codes, self.stock_names = self._load_stock_list()
        self.state = self._load_state()

        # 行业缓存：code -> (industry_name, industry_level)
        self.stock_industry = {}
        self.industry_stocks = {}
        self.industry_size = {}

    # -------------------- 股票列表加载 --------------------
    def _load_stock_list(self) -> Tuple[List[str], Dict[str, str]]:
        try:
            df = pd.read_csv(self.csv_path, encoding='utf-8')
            if self.code_column not in df.columns:
                raise ValueError(f"CSV中未找到列 '{self.code_column}'")
            codes = df[self.code_column].astype(str).str.strip().tolist()
            formatted_codes = []
            for c in codes:
                if not c.startswith(('sh.', 'sz.', 'bj.')):
                    if c.startswith('6'):
                        c = 'sh.' + c
                    elif c.startswith(('0', '3')):
                        c = 'sz.' + c
                    else:
                        c = 'sh.' + c
                formatted_codes.append(c)

            if self.name_column in df.columns:
                name_dict = dict(zip(formatted_codes, df[self.name_column].astype(str)))
            else:
                name_dict = {c: '未知' for c in formatted_codes}
            logger.info(f"从CSV加载 {len(formatted_codes)} 只股票")
            return formatted_codes, name_dict
        except Exception as e:
            logger.error(f"股票列表加载失败: {e}")
            return [], {}

    # -------------------- 状态持久化 --------------------
    def _load_state(self) -> dict:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'rb') as f:
                    state = pickle.load(f)
                logger.info("📂 加载历史状态成功")
                return state
            except Exception:
                pass
        return {
            'prev_main_sector': None,
            'prev_hl_stocks': [],
            'prev_L2': None,
            'prev_drop_ratio_trigger': False,
            'leader_code': None,
            'leader_break_days': 0,
            'daily_fry_rate_seq': [],
            'daily_promote_rate_seq': [],
            'stock_burst_count': {}
        }

    def _save_state(self):
        with open(self.state_file, 'wb') as f:
            pickle.dump(self.state, f)

    # -------------------- 登录/登出 --------------------
    def login(self) -> bool:
        for attempt in range(self.max_retries):
            try:
                time.sleep(self.request_delay)
                lg = bs.login()
                if lg.error_code == '0':
                    self.is_logged_in = True
                    logger.info("✅ 登录成功")
                    return True
            except:
                pass
            time.sleep(2 ** attempt)
        return False

    def logout(self):
        if self.is_logged_in:
            try:
                bs.logout()
            except:
                pass
            self.is_logged_in = False
            logger.info("✅ 已登出")

    # -------------------- 交易日校验 --------------------
    @staticmethod
    def _is_trade_date(date_str: str) -> bool:
        """检查日期是否为交易日（通过 baostock 查询）"""
        try:
            rs = bs.query_trade_dates(start_date=date_str, end_date=date_str)
            if rs.error_code == '0':
                while rs.next():
                    row = rs.get_row_data()
                    if row[1] == '1':  # is_trading_day
                        return True
        except:
            pass
        return False

    def _validate_date(self, date_str: str):
        """校验分析日期，若无效则报错并提示最近交易日"""
        today_str = datetime.now().strftime('%Y-%m-%d')
        if date_str > today_str:
            logger.error(f"分析日期 {date_str} 晚于今天，无法获取数据")
            # 获取最近一个交易日
            try:
                rs = bs.query_trade_dates(start_date='2020-01-01', end_date=today_str)
                last_trade = None
                while rs.next():
                    row = rs.get_row_data()
                    if row[1] == '1':
                        last_trade = row[0]
                if last_trade:
                    logger.info(f"建议使用最近交易日: {last_trade}")
            except:
                pass
            raise ValueError(f"日期 {date_str} 无效，请使用今天或之前的交易日")

        if not self._is_trade_date(date_str):
            logger.warning(f"{date_str} 不是交易日，将尝试获取数据（可能为空）")

    # -------------------- 行业缓存初始化（增强版） --------------------
    def _init_industry_cache(self):
        """获取每只股票的申万行业，优先二级，无二级时回退一级"""
        logger.info("开始初始化行业缓存...")
        self.stock_industry.clear()
        self.industry_stocks.clear()
        unknown_count = 0
        for code in tqdm(self.stock_codes, desc='行业缓存'):
            try:
                time.sleep(0.05)
                rs = bs.query_stock_industry(code)
                if rs.error_code != '0':
                    unknown_count += 1
                    continue
                industries = []
                while rs.next():
                    row = rs.get_row_data()
                    ind_name = row[1]
                    ind_class = row[2] if len(row) > 2 else ''
                    industries.append((ind_name, ind_class))
                # 优先选择二级行业
                selected = None
                for name, cls in industries:
                    if '二级' in cls:
                        selected = name
                        break
                if not selected:
                    # 回退到一级行业
                    for name, cls in industries:
                        if '一级' in cls:
                            selected = name
                            break
                if not selected and industries:
                    # 使用第一个行业
                    selected = industries[0][0]
                if selected:
                    self.stock_industry[code] = selected
                    self.industry_stocks.setdefault(selected, []).append(code)
                else:
                    unknown_count += 1
            except Exception as e:
                unknown_count += 1
                continue
        self.industry_size = {k: len(v) for k, v in self.industry_stocks.items()}
        logger.info(f"行业缓存完成：共识别 {len(self.stock_industry)}/{len(self.stock_codes)} 只股票的行业，"
                    f"覆盖 {len(self.industry_stocks)} 个行业")
        if unknown_count > 0:
            logger.warning(f"有 {unknown_count} 只股票未能识别行业")
        # 输出样例
        sample_codes = list(self.stock_industry.keys())[:5]
        if sample_codes:
            logger.info("行业映射样例：")
            for c in sample_codes:
                logger.info(f"  {c} -> {self.stock_industry[c]}")

    # -------------------- 数据获取（增强异常处理） --------------------
    def get_daily_data(self, code: str, end_date: str, days: int = 150) -> Optional[pd.DataFrame]:
        start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=days * 2)).strftime('%Y-%m-%d')
        fields = "date,code,open,high,low,close,preclose,volume,amount,turn"
        for attempt in range(self.max_retries + 1):
            try:
                delay = self.request_delay + (2 ** attempt) + np.random.uniform(0, 1)
                time.sleep(delay)
                rs = bs.query_history_k_data_plus(
                    code, fields,
                    start_date=start_date, end_date=end_date,
                    frequency="d", adjustflag="3"
                )
                if rs.error_code != '0':
                    continue
                data_list = []
                while (rs.error_code == '0') & rs.next():
                    try:
                        data_list.append(rs.get_row_data())
                    except (UnicodeDecodeError, Exception) as decode_err:
                        logger.warning(f"[{code}] 解析行数据异常: {decode_err}，重试")
                        data_list = []
                        break
                if not data_list and attempt < self.max_retries:
                    continue
                if not data_list:
                    return None
                df = pd.DataFrame(data_list, columns=fields.split(','))
                for col in ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turn']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df['date'] = pd.to_datetime(df['date'])
                df = df.dropna(subset=['date', 'close', 'preclose'])
                df = df.sort_values('date').reset_index(drop=True)
                if len(df) > days:
                    df = df.tail(days)
                return df
            except Exception as e:
                err_msg = str(e).lower()
                if any(kw in err_msg for kw in ['decode', 'decompress', 'zlib', 'invalid']):
                    logger.debug(f"[{code}] 数据异常重试: {e}")
                    time.sleep(1)
                    continue
                logger.error(f"[{code}] 未处理异常: {e}")
                return None
        return None

    # -------------------- 涨停/跌停/连板判断 --------------------
    @staticmethod
    def is_limit_up(row) -> bool:
        if row['preclose'] <= 0:
            return False
        up_ratio = row['close'] / row['preclose'] - 1
        if up_ratio < 0.095:
            return False
        if row['open'] == row['high'] == row['close'] and row['close'] == row['high']:
            if pd.notna(row['turn']) and row['turn'] < 1.0:
                return False
        return True

    @staticmethod
    def is_limit_down(row) -> bool:
        if row['preclose'] <= 0:
            return False
        down_ratio = row['close'] / row['preclose'] - 1
        return down_ratio <= -0.095

    @staticmethod
    def is_break_up(row) -> bool:
        if row['preclose'] <= 0:
            return False
        return (row['high'] / row['preclose'] - 1 >= 0.095) and (row['close'] < row['high'])

    def count_consecutive_boards(self, df: pd.DataFrame, date_idx: int) -> int:
        count = 0
        for i in range(date_idx, -1, -1):
            row = df.iloc[i]
            if i == date_idx:
                if self.is_limit_up(row):
                    count += 1
                else:
                    break
            else:
                if self.is_limit_up(row):
                    count += 1
                else:
                    break
        return count

    def get_20d_burst_count(self, code: str, end_date: str) -> int:
        df = self.get_daily_data(code, end_date, days=30)
        if df is None:
            return 0
        df = df.tail(20)
        cnt = 0
        for _, row in df.iterrows():
            if self.is_limit_up(row):
                cnt += 1
        return cnt

    # -------------------- 单只股票分析（线程安全） --------------------
    def _analyze_single_stock(self, code: str, date_str: str, burst_cache: dict) -> dict:
        result = {
            'code': code,
            'hl_status': 'none',
            'limit_down': False,
            'high_face': False,
            'sky_floor': False,
            'board_num': 0,
            'turn': 0.0,
            'preclose': 0.0,
            'high': 0.0,
            'low': 0.0,
            'close': 0.0
        }
        try:
            df = self.get_daily_data(code, date_str, days=70)
            if df is None or df.empty:
                return result
            latest = df.iloc[-1]
            result['preclose'] = latest['preclose']
            result['high'] = latest['high']
            result['low'] = latest['low']
            result['close'] = latest['close']
            result['turn'] = latest['turn'] if pd.notna(latest['turn']) else 0.0

            if self.is_limit_down(latest):
                result['limit_down'] = True

            if self.is_limit_up(latest):
                result['hl_status'] = 'limit_up'
                result['board_num'] = self.count_consecutive_boards(df, len(df) - 1)
            elif self.is_break_up(latest):
                result['hl_status'] = 'break_up'
                result['board_num'] = self.count_consecutive_boards(df, len(df) - 1)

            if code not in burst_cache:
                burst_cache[code] = self.get_20d_burst_count(code, date_str)
            burst_count = burst_cache.get(code, 0)
            if burst_count >= 5 and latest['preclose'] > 0:
                pullback = (latest['high'] - latest['close']) / latest['preclose']
                if pullback >= 0.08:
                    result['high_face'] = True
                    if latest['close'] / latest['preclose'] - 1 <= -0.09 and \
                            latest['high'] / latest['preclose'] - 1 >= 0.09:
                        result['sky_floor'] = True
        except Exception as e:
            logger.debug(f"股票 {code} 分析异常: {e}")
        return result

    # -------------------- 主线行业识别 --------------------
    def detect_main_sector(self, hl_stocks_info: List[dict]):
        if not hl_stocks_info:
            return {'has_main': False, 'sector_name': None, 'hl_count': 0}

        industry_hl = {}
        for stock in hl_stocks_info:
            ind = self.stock_industry.get(stock['code'])
            if ind:
                industry_hl[ind] = industry_hl.get(ind, 0) + 1

        if not industry_hl:
            logger.warning("涨停股均无行业映射，无法识别主线")
            return {'has_main': False, 'sector_name': None, 'hl_count': 0}

        max_ind = max(industry_hl, key=industry_hl.get)
        max_count = industry_hl[max_ind]

        prev = self.state.get('prev_main_sector')
        if prev and prev.get('sector_name'):
            prev_ind = prev['sector_name']
            prev_count = industry_hl.get(prev_ind, 0)
            threshold = max(3, max_count * 0.5)
            if prev_count >= threshold:
                return {'has_main': True, 'sector_name': prev_ind, 'hl_count': prev_count}

        return {'has_main': True, 'sector_name': max_ind, 'hl_count': max_count}

    # -------------------- L1 --------------------
    def calc_L1(self, hs300_df: pd.DataFrame, limit_up_count: int,
                limit_down_count: int, high_face_count: int) -> Tuple[float, bool, bool]:
        score = 10.0
        systemic_risk = False
        if hs300_df is not None and not hs300_df.empty:
            chg = hs300_df.iloc[-1]['close'] / hs300_df.iloc[-1]['preclose'] - 1
            if chg <= -0.025:
                score -= 4
                systemic_risk = True
        drop_trigger = False
        if limit_up_count > 0 and (limit_down_count / limit_up_count) > 1.2:
            drop_trigger = True
            if self.state.get('prev_drop_ratio_trigger', False):
                score -= 3
        if high_face_count >= 15:
            score -= 3
        elif high_face_count >= 8:
            score -= 1
        score = max(0, min(10, score))
        return score, systemic_risk, drop_trigger

    # -------------------- L2 --------------------
    def calc_L2(self, main_industry_name: str, hl_stocks_info: List[dict],
                high_face_count: int) -> float:
        if not main_industry_name:
            return 0.0
        sector_codes = self.industry_stocks.get(main_industry_name, [])
        if not sector_codes:
            return 0.0
        main_hl = [s for s in hl_stocks_info if s['code'] in sector_codes]

        board_4_plus = len([s for s in main_hl if s.get('board_num', 0) >= 4])
        tier_score = 4 if board_4_plus >= 3 else (2 if board_4_plus >= 1 else 0)

        fry_rates = self.state.get('daily_fry_rate_seq', [])
        avg_fry = np.mean(fry_rates[-20:]) if len(fry_rates) >= 20 else 0.3
        high_board = [s for s in main_hl if s.get('board_num', 0) >= 3 and s.get('is_break')]
        valid_high = [s for s in main_hl if s.get('board_num', 0) >= 3]
        today_fry = len(high_board) / len(valid_high) if valid_high else 0
        diff = today_fry - avg_fry
        fry_score = 4 if diff < 0 else (2 if diff <= 0.15 else 0)

        promote_rates = self.state.get('daily_promote_rate_seq', [])
        avg_promote = np.mean(promote_rates[-20:]) if len(promote_rates) >= 20 else 0.3
        yesterday_hl = self.state.get('prev_hl_stocks', [])
        today_hl_codes = {s['code'] for s in hl_stocks_info if not s.get('is_break')}
        promote_count = len([c for c in yesterday_hl if c in today_hl_codes])
        today_promote = promote_count / len(yesterday_hl) if yesterday_hl else 0
        dev = today_promote - avg_promote
        promote_score = 2 if dev >= 0 else (1 if dev >= -0.20 else 0)

        if high_face_count == 0:
            face_score = 3
        elif 1 <= high_face_count <= 5:
            face_score = 1
        else:
            has_sky = any(s.get('is_sky_floor', False) for s in hl_stocks_info)
            face_score = -1 if (high_face_count >= 6 and has_sky) else 0

        L2 = tier_score + fry_score + promote_score + face_score
        return min(13, L2)

    # -------------------- L3 --------------------
    def detect_leader(self, main_industry_name: str, hl_stocks_info: List[dict]) -> Optional[str]:
        if not main_industry_name:
            return None
        sector_codes = self.industry_stocks.get(main_industry_name, [])
        candidates = [s for s in hl_stocks_info if s['code'] in sector_codes and not s.get('is_break')]
        if not candidates:
            return None
        candidates.sort(key=lambda x: (-x.get('board_num', 0), -x.get('turn', 0)))
        return candidates[0]['code']

    def calc_L3_signal(self, leader_code: str, today_hl_codes: set,
                       hl_stocks_info: List[dict]) -> Tuple[str, int]:
        prev_leader = self.state.get('leader_code')
        break_days = self.state.get('leader_break_days', 0)
        if leader_code != prev_leader and prev_leader is not None:
            break_days = 0
        leader_board = leader_code in today_hl_codes if leader_code else False
        new_break_days = 0 if leader_board else break_days + 1
        sky_floor_cnt = sum(1 for s in hl_stocks_info if s.get('is_sky_floor', False))

        if new_break_days >= 2 or sky_floor_cnt >= 2:
            signal = 'confirm'
        elif new_break_days >= 1 or sky_floor_cnt >= 1:
            signal = 'warning'
        else:
            signal = 'none'
        return signal, new_break_days

    # -------------------- 动态阈值与仓位 --------------------
    @staticmethod
    def dynamic_params(L1_score: float):
        if L1_score <= 4:
            return '恐慌', {'止损': '-3%', '竞价溢价': '+1%', '跌停容量': 5, '量比阈值': 0.03, '入场最低分': 38}
        elif L1_score <= 7:
            return '中性', {'止损': '-5%', '竞价溢价': '+1.5%', '跌停容量': 10, '量比阈值': 0.05, '入场最低分': 45}
        else:
            return '亢奋', {'止损': '-7%', '竞价溢价': '+2%', '跌停容量': 15, '量比阈值': 0.07, '入场最低分': 55}

    @staticmethod
    def position_advice(L2_score: float, systemic_risk: bool):
        if systemic_risk:
            return 0, 0, '0档（空仓）'
        if L2_score >= 16:
            return 0.25, 0.9, '5档'
        elif L2_score >= 12:
            return 0.20, 0.7, '4档'
        elif L2_score >= 8:
            return 0.10, 0.3, '3档'
        elif L2_score >= 4:
            return 0.05, 0.1, '2档'
        else:
            return 0, 0, '1档（空仓）'

    # -------------------- 主分析函数 --------------------
    def analyze(self, date_str: str):
        logger.info(f"======== 开始分析 {date_str} ========")
        if not self.stock_codes:
            logger.error("股票列表为空，请检查CSV文件")
            return

        # 日期校验（需要先确保登录状态）
        if self.is_logged_in:
            try:
                self._validate_date(date_str)
            except ValueError as e:
                logger.error(str(e))
                return
        else:
            logger.warning("未登录，无法校验日期，继续尝试获取数据")

        # 初始化行业缓存
        if not self.stock_industry:
            self._init_industry_cache()

        # 沪深300数据
        hs300_df = self.get_daily_data('sh.000300', date_str, days=5)

        # ---------- 扫描股票 ----------
        burst_cache = self.state.get('stock_burst_count', {}).copy()
        lock = threading.Lock() if self.max_workers > 1 else None
        limit_up_list = []
        limit_down_count = 0
        high_face_count = 0
        today_hl_codes = set()

        if self.max_workers <= 1:
            logger.info("单线程扫描模式")
            for code in tqdm(self.stock_codes, desc="个股扫描"):
                res = self._analyze_single_stock(code, date_str, burst_cache)
                if res is None:
                    continue
                if res.get('limit_down'):
                    limit_down_count += 1
                if res.get('hl_status') in ('limit_up', 'break_up'):
                    info = {
                        'code': code,
                        'board_num': res.get('board_num', 0),
                        'turn': res.get('turn', 0),
                        'is_break': res.get('hl_status') == 'break_up',
                        'is_sky_floor': res.get('sky_floor', False),
                        'preclose': res.get('preclose', 0),
                        'high': res.get('high', 0),
                        'low': res.get('low', 0),
                        'close': res.get('close', 0)
                    }
                    limit_up_list.append(info)
                    if not info['is_break']:
                        today_hl_codes.add(code)
                if res.get('high_face'):
                    high_face_count += 1
        else:
            logger.info(f"多线程扫描模式 ({self.max_workers} 线程)")
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_code = {
                    executor.submit(self._analyze_single_stock, code, date_str, burst_cache): code
                    for code in self.stock_codes
                }
                with tqdm(total=len(self.stock_codes), desc="个股扫描") as pbar:
                    for future in as_completed(future_to_code):
                        code = future_to_code[future]
                        try:
                            res = future.result()
                        except:
                            res = None
                        if res is None:
                            pbar.update(1)
                            continue
                        if res.get('limit_down'):
                            with lock: limit_down_count += 1
                        if res.get('hl_status') in ('limit_up', 'break_up'):
                            info = {
                                'code': code,
                                'board_num': res.get('board_num', 0),
                                'turn': res.get('turn', 0),
                                'is_break': res.get('hl_status') == 'break_up',
                                'is_sky_floor': res.get('sky_floor', False),
                                'preclose': res.get('preclose', 0),
                                'high': res.get('high', 0),
                                'low': res.get('low', 0),
                                'close': res.get('close', 0)
                            }
                            with lock:
                                limit_up_list.append(info)
                                if not info['is_break']:
                                    today_hl_codes.add(code)
                        if res.get('high_face'):
                            with lock: high_face_count += 1
                        pbar.update(1)

        self.state['stock_burst_count'] = burst_cache
        logger.info(f"扫描完成：涨停(含炸板)={len(limit_up_list)}, 跌停={limit_down_count}, "
                    f"高位大面={high_face_count}")

        if len(limit_up_list) == 0:
            logger.warning("⚠️ 当日无任何涨停/炸板股！可能原因：非交易日、股票池过小、行情极端")

        # ---------- 主线识别 ----------
        main_info = self.detect_main_sector(limit_up_list)
        logger.info(f"主线行业: {main_info}")

        # ---------- L1/L2/L3 ----------
        L1_score, systemic_risk, drop_trigger = self.calc_L1(
            hs300_df, len(today_hl_codes), limit_down_count, high_face_count
        )
        L2_score = self.calc_L2(main_info.get('sector_name'), limit_up_list, high_face_count)
        leader_code = self.detect_leader(main_info.get('sector_name'), limit_up_list)
        L3_signal, break_days = self.calc_L3_signal(leader_code, today_hl_codes, limit_up_list)

        prev_L2 = self.state.get('prev_L2')
        tide_confirm = False
        if prev_L2 is not None:
            if (L2_score <= 6) and (prev_L2 <= 7) and (L3_signal in ('warning', 'confirm')):
                tide_confirm = True

        mood, thresholds = self.dynamic_params(L1_score)
        single_pos, total_pos, pos_tier = self.position_advice(L2_score, systemic_risk)

        self.state['prev_main_sector'] = {
            'name': main_info.get('sector_name'),
            'hl_count': main_info.get('hl_count', 0)
        }
        self.state['prev_hl_stocks'] = list(today_hl_codes)
        self.state['prev_L2'] = L2_score
        self.state['prev_drop_ratio_trigger'] = drop_trigger
        self.state['leader_code'] = leader_code
        self.state['leader_break_days'] = break_days
        self._save_state()

        # ---------- 报告 ----------
        report = f"""
        ==============================================
               情绪周期日度分析报告 ({date_str})
        ==============================================
        【L1 全局市场情绪】 {L1_score:.1f} 分  环境档位: {mood}
        【L2 主线情绪】 {L2_score:.1f} 分  (满分13，无竞价)
        【L3 龙头信号】 {L3_signal}  (龙头: {leader_code}, 断板 {break_days}天)
        【双日退潮确认】 {'🔴 已确认！' if tide_confirm else '🟢 未触发'}
        ----------------------------------------------
        【动态阈值】
          止损: {thresholds['止损']}  竞价溢价: {thresholds['竞价溢价']}
          跌停容量: {thresholds['跌停容量']}  量比阈值: {thresholds['量比阈值']}
          入场最低分: {thresholds['入场最低分']}
        【仓位建议】
          单股上限: {single_pos * 100:.0f}%  总仓上限: {total_pos * 100:.0f}%  ({pos_tier})
        ----------------------------------------------
        【主线行业】 {'有' if main_info['has_main'] else '无'}
          行业: {main_info.get('sector_name')}
          涨停数: {main_info.get('hl_count')}
        """
        if tide_confirm:
            report += "\n        ⚠️ 双日退潮确认！建议清仓禁买！\n"
        report += "=============================================="
        print(report)
        logger.info(report)
        return {
            'L1': L1_score, 'L2': L2_score, 'L3': L3_signal,
            'tide_confirm': tide_confirm, 'mood': mood,
            'single_pos': single_pos, 'total_pos': total_pos
        }

    def run(self, date_str: str = None):
        if date_str is None:
            date_str = datetime.now().strftime('%Y-%m-%d')
        print("=" * 60)
        print("情绪周期日度分析器（行业增强版）")
        print(f"股票池: {self.csv_path} ({len(self.stock_codes)}只)")
        print(f"分析日期: {date_str}")
        print("=" * 60)
        if not self.login():
            logger.error("登录失败")
            return
        try:
            self.analyze(date_str)
        except Exception as e:
            logger.error(f"分析异常: {traceback.format_exc()}")
        finally:
            self.logout()


# -------------------- 示例CSV生成 --------------------
def create_sample_csv(filepath='stock_pool.csv'):
    sample = {
        'code': ['600519', '000858', '300750', '002594', '601012'],
        'name': ['贵州茅台', '五粮液', '宁德时代', '比亚迪', '隆基绿能']
    }
    pd.DataFrame(sample).to_csv(filepath, index=False, encoding='utf-8-sig')
    logger.info(f"示例股票池已生成: {filepath}")


def main():
    print("=" * 60)
    print("情绪周期日度分析器（行业增强版）")
    print("=" * 60)
    csv_path = input("请输入股票池CSV路径: ").strip()
    if not os.path.exists(csv_path):
        print("文件不存在，是否创建示例？(y/n)")
        if input().strip().lower() == 'y':
            create_sample_csv(csv_path)
        else:
            return

    code_col = input("请输入代码列名 (默认'code'): ").strip() or 'code'
    name_col = input("请输入名称列名 (默认'name'): ").strip() or 'name'
    date_input = input("请输入分析日期 (YYYY-MM-DD，回车为今天): ").strip()
    workers = input("请输入并发线程数 (建议1，稳定后可尝试2，默认1): ").strip()
    max_workers = int(workers) if workers.isdigit() else 1

    analyzer = EmotionCycleAnalyzer(
        csv_path=csv_path,
        code_column=code_col,
        name_column=name_col,
        state_file='emotion_state_industry.pkl',
        max_workers=max_workers
    )
    analyzer.run(date_input if date_input else None)


if __name__ == "__main__":
    main()