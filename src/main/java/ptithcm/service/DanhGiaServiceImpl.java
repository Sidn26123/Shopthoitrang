package ptithcm.service;

import java.util.List;

import javax.transaction.Transactional;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import ptithcm.entity.DanhGiaEntity;
import ptithcm.dao.DanhGiaDao;

@Service
@Transactional
public class DanhGiaServiceImpl implements DanhGiaService {
	
	@Autowired
	DanhGiaDao danhGiaDAO;
	
	@Override
	public List<DanhGiaEntity> layDanhGiaSanPham(String maSp) {
		return danhGiaDAO.layDanhGiaSanPham(maSp);
	}

	@Override
	public void saveDanhGia(DanhGiaEntity danhGia) {
		danhGiaDAO.saveDanhGia(danhGia);
		
	}
	@Override
	public List<DanhGiaEntity> layDanhGiaSanPhamTheoMaND(int maND) {
		return danhGiaDAO.layDanhGiaSanPhamTheoMaND(maND);
		
	}

}
