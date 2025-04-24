package ptithcm.service;

import java.util.List;

import javax.transaction.Transactional;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import ptithcm.entity.SanPhamEntity;
import ptithcm.entity.KieuSanPhamEntity;
import ptithcm.dao.KieuSanPhamDAO;

@Service
@Transactional
public class KieuSanPhamServiceImpl implements KieuSanPhamService{
	@Autowired
	KieuSanPhamDAO kieuSanPhamDAO;
	@Autowired
	SanPhamService sanPhamService;

	@Override
	public List<KieuSanPhamEntity> layKieu() {
		return kieuSanPhamDAO.layKieu();
	}

	@Override
	public KieuSanPhamEntity layKieuTheoMa(int maKieu) {
		return kieuSanPhamDAO.layKieuTheoMa(maKieu);
	}
	
	@Override
	public boolean kiemTraSanPhamTheoKieu(int maKieu) {
	    List<SanPhamEntity> danhSachSanPham = sanPhamService.layAllSanPham();
	    
	    for (SanPhamEntity sanPham : danhSachSanPham) {
	        KieuSanPhamEntity kieuSanPham = sanPham.getMaKieu();
	        
	        if (kieuSanPham != null && kieuSanPham.getMaKieu() == maKieu) {
	            return true; // Có sản phẩm liên kết với mã kiểu đã cho
	        }
	    }
	    
	    return false; // Không có sản phẩm liên kết với mã kiểu đã cho
	}

	

	@Override
	public void themKieu(KieuSanPhamEntity kieu) {
		kieuSanPhamDAO.themKieu(kieu);
	}

	@Override
	public void updateKieu(KieuSanPhamEntity kieu) {
		kieuSanPhamDAO.updateKieu(kieu);
	}

	@Override
	public void xoaKieu(KieuSanPhamEntity kieu) {
		kieuSanPhamDAO.xoaKieu(kieu);
	}

	
}
